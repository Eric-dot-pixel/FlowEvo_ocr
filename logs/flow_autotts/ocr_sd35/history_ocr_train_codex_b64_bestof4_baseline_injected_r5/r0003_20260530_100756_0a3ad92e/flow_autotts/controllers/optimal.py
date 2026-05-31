"""Candidate controller for SD3.5 OCR discovery."""

from __future__ import annotations

from flow_autotts.core.env import FlowTTSEnv
from flow_autotts.core.errors import BudgetExceededError, InvalidActionError
from flow_autotts.core.state import AnswerRecord, PreviewRecord


class OptimalController:
    """Preview-guided controller with adaptive refinement and target-NFE discipline."""

    def solve(self, env: FlowTTSEnv, beta: float) -> AnswerRecord:
        beta = min(max(float(beta), 0.0), 1.0)
        schedule = self._schedule(env, beta)
        initial_budget = max(0, int(env.budget_left))
        target_nfe = min(initial_budget, int(schedule["target_nfe"]))

        try:
            root_ids = env.spawn(int(schedule["root_count"]))
        except InvalidActionError:
            return self._safe_answer(env)

        affordable = self._affordable_roots(env, target_nfe, initial_budget, schedule)
        root_ids = root_ids[:affordable]
        if not root_ids:
            return self._safe_answer(env)

        self._scout_roots(env, root_ids, schedule, target_nfe, initial_budget)
        self._prune_after_scout(env, schedule)
        self._finish_survivors(env, schedule, target_nfe, initial_budget)
        self._confirm_top_anchors(env, schedule, target_nfe, initial_budget)
        self._refine_best_anchors(env, schedule, target_nfe, initial_budget)
        self._tail_confirm_or_refine(env, schedule, target_nfe, initial_budget)
        return self._safe_answer(env)

    def _schedule(self, env: FlowTTSEnv, beta: float) -> dict[str, float | int | bool | str]:
        target_nfe = self._beta_target_nfe(beta)
        budget_cap = min(int(env.budget), 64)
        target_nfe = min(budget_cap, target_nfe)

        if beta <= 0.0:
            return {
                "target_nfe": min(budget_cap, 10),
                "root_count": 2,
                "scout_time": 0.62,
                "survivor_count": 1,
                "confirm_count": 1,
                "confirm_steps": 0,
                "backward_children": 0,
                "backward_time": 0.88,
                "noise_policy": "fresh_noise",
                "noise_strength": 1.0,
                "prune_margin": 0.32,
                "uncertainty_gate": 0.50,
                "late_preview_rounds": 1,
                "use_sde_scout": False,
            }

        return {
            "target_nfe": target_nfe,
            "root_count": 2 + int(2 * beta) + (1 if beta >= 0.75 else 0),
            "scout_time": 0.38 + 0.14 * beta,
            "survivor_count": 1 + int(2 * beta),
            "confirm_count": 1 + int(2 * beta),
            "confirm_steps": 1 + int(2 * beta),
            "backward_children": 1 + int(3 * beta),
            "backward_time": 0.84 - 0.16 * beta,
            "noise_policy": "mixed_noise" if beta >= 0.70 else "fresh_noise",
            "noise_strength": 0.55 if beta >= 0.70 else 1.0,
            "prune_margin": 0.16 - 0.06 * beta,
            "uncertainty_gate": 0.48 - 0.16 * beta,
            "late_preview_rounds": 1 + int(3 * beta),
            "use_sde_scout": beta >= 0.75,
        }

    def _beta_target_nfe(self, beta: float) -> int:
        if beta <= 0.0:
            return 10
        if beta <= 0.25:
            return 20
        if beta <= 0.50:
            return 36
        if beta <= 0.75:
            return 48
        return 64

    def _affordable_roots(
        self,
        env: FlowTTSEnv,
        target_nfe: int,
        initial_budget: int,
        schedule: dict[str, float | int | bool | str],
    ) -> int:
        scout_time = float(schedule["scout_time"])
        scout_cost = max(1, self._steps_from_zero(env, scout_time)) + 1
        remaining = max(1, target_nfe - self._spent(env, initial_budget))
        return max(1, min(int(schedule["root_count"]), remaining // max(1, scout_cost)))

    def _scout_roots(
        self,
        env: FlowTTSEnv,
        root_ids: list[int],
        schedule: dict[str, float | int | bool | str],
        target_nfe: int,
        initial_budget: int,
    ) -> None:
        scout_time = float(schedule["scout_time"])
        solver = "sde" if bool(schedule["use_sde_scout"]) else "euler"
        cfg = {"noise_scale": 0.008, "sigma_max": 1.15, "min_time": 0.02} if solver == "sde" else None
        for particle_id in root_ids:
            cost = self._move_cost(env, particle_id, scout_time) + 1
            if not self._can_afford(env, target_nfe, initial_budget, cost):
                return
            try:
                self._forward_to(env, particle_id, scout_time, solver=solver, cfg=cfg)
                self._preview(env, particle_id, target_nfe, initial_budget)
            except (BudgetExceededError, InvalidActionError):
                return

    def _prune_after_scout(
        self,
        env: FlowTTSEnv,
        schedule: dict[str, float | int | bool | str],
    ) -> None:
        previews = self._ranked_previews(env)
        if not previews:
            return

        best = float(previews[0].score or 0.0)
        keep_ids: list[int] = []
        for index, preview in enumerate(previews):
            gap = best - float(preview.score or 0.0)
            uncertain = float(preview.uncertainty or 0.0) > float(schedule["uncertainty_gate"])
            if index < int(schedule["survivor_count"]) or gap <= float(schedule["prune_margin"]) or uncertain:
                keep_ids.append(preview.particle_id)

        prune_ids = [
            preview.particle_id
            for preview in previews
            if preview.particle_id not in keep_ids and self._is_active(env, preview.particle_id)
        ]
        if prune_ids:
            try:
                env.prune(prune_ids)
            except InvalidActionError:
                return

    def _finish_survivors(
        self,
        env: FlowTTSEnv,
        schedule: dict[str, float | int | bool | str],
        target_nfe: int,
        initial_budget: int,
    ) -> None:
        previews = self._ranked_previews(env)
        survivors = previews[: int(schedule["survivor_count"])]
        for preview in survivors:
            pid = preview.particle_id
            if not self._is_active(env, pid):
                continue
            need_preview = self._needs_confirmation(preview, schedule)
            cost = self._move_cost(env, pid, 1.0) + (1 if need_preview else 0)
            if not self._can_afford(env, target_nfe, initial_budget, cost):
                continue
            try:
                self._forward_to(env, pid, 1.0, solver="euler", cfg=None)
                if need_preview:
                    self._preview(env, pid, target_nfe, initial_budget)
            except (BudgetExceededError, InvalidActionError):
                continue

    def _confirm_top_anchors(
        self,
        env: FlowTTSEnv,
        schedule: dict[str, float | int | bool | str],
        target_nfe: int,
        initial_budget: int,
    ) -> None:
        previews = self._ranked_previews(env)[: int(schedule["confirm_count"])]
        confirm_steps = int(schedule["confirm_steps"])
        if confirm_steps <= 0:
            return

        for preview in previews:
            pid = preview.particle_id
            if not self._is_active(env, pid):
                continue
            if not self._needs_confirmation(preview, schedule):
                continue
            next_time = self._advance_time(env, pid, confirm_steps)
            if next_time is None:
                continue
            cost = self._move_cost(env, pid, next_time) + 1
            if not self._can_afford(env, target_nfe, initial_budget, cost):
                continue
            try:
                self._forward_to(env, pid, next_time, solver="euler", cfg=None)
                self._preview(env, pid, target_nfe, initial_budget)
            except (BudgetExceededError, InvalidActionError):
                continue

    def _refine_best_anchors(
        self,
        env: FlowTTSEnv,
        schedule: dict[str, float | int | bool | str],
        target_nfe: int,
        initial_budget: int,
    ) -> None:
        budgeted_children = int(schedule["backward_children"])
        if budgeted_children <= 0:
            return

        launched = 0
        for anchor in self._refinement_candidates(env):
            if launched >= budgeted_children:
                return
            child_count = min(
                budgeted_children - launched,
                self._children_for_anchor(env, anchor, schedule),
            )
            if child_count <= 0:
                continue
            per_child_cost = self._child_cost(env, float(schedule["backward_time"]))
            total_cost = child_count * per_child_cost
            if not self._can_afford(env, target_nfe, initial_budget, total_cost):
                child_count = max(0, (target_nfe - self._spent(env, initial_budget)) // max(1, per_child_cost))
            if child_count <= 0:
                return
            try:
                child_ids = env.backward(
                    anchor.id,
                    target_time=float(schedule["backward_time"]),
                    noise_policy=str(schedule["noise_policy"]),
                    num_children=child_count,
                    strength=float(schedule["noise_strength"]),
                )
            except (BudgetExceededError, InvalidActionError):
                continue

            for child_id in child_ids:
                if launched >= budgeted_children:
                    return
                if not self._can_afford(env, target_nfe, initial_budget, per_child_cost):
                    return
                try:
                    self._forward_to(env, child_id, 1.0, solver="euler", cfg=None)
                    self._preview(env, child_id, target_nfe, initial_budget)
                    launched += 1
                except (BudgetExceededError, InvalidActionError):
                    return

    def _tail_confirm_or_refine(
        self,
        env: FlowTTSEnv,
        schedule: dict[str, float | int | bool | str],
        target_nfe: int,
        initial_budget: int,
    ) -> None:
        rounds = int(schedule["late_preview_rounds"])
        while rounds > 0 and self._spent(env, initial_budget) < target_nfe and env.budget_left > 0:
            best = self._best_live_preview(env)
            if best is None:
                return

            pid = best.particle_id
            next_time = self._advance_time(env, pid, 1) if self._is_active(env, pid) else None
            if next_time is not None:
                confirm_cost = self._move_cost(env, pid, next_time) + 1
                if self._can_afford(env, target_nfe, initial_budget, confirm_cost):
                    try:
                        self._forward_to(env, pid, next_time, solver="euler", cfg=None)
                        self._preview(env, pid, target_nfe, initial_budget)
                        rounds -= 1
                        continue
                    except (BudgetExceededError, InvalidActionError):
                        return

            child_cost = self._child_cost(env, float(schedule["backward_time"]))
            if self._can_afford(env, target_nfe, initial_budget, child_cost):
                try:
                    child_ids = env.backward(
                        best.id,
                        target_time=float(schedule["backward_time"]),
                        noise_policy=str(schedule["noise_policy"]),
                        num_children=1,
                        strength=float(schedule["noise_strength"]),
                    )
                except (BudgetExceededError, InvalidActionError):
                    return
                if not child_ids:
                    return
                try:
                    self._forward_to(env, child_ids[0], 1.0, solver="euler", cfg=None)
                    self._preview(env, child_ids[0], target_nfe, initial_budget)
                    rounds -= 1
                    continue
                except (BudgetExceededError, InvalidActionError):
                    return

            if self._can_afford(env, target_nfe, initial_budget, 1):
                try:
                    env.preview(pid, mode="clean_anchor", scorer="default")
                    rounds -= 1
                    continue
                except (BudgetExceededError, InvalidActionError):
                    return
            return

        while self._spent(env, initial_budget) + self._child_cost(env, float(schedule["backward_time"])) <= target_nfe:
            best = self._best_live_preview(env)
            if best is None:
                return
            try:
                child_ids = env.backward(
                    best.id,
                    target_time=float(schedule["backward_time"]),
                    noise_policy=str(schedule["noise_policy"]),
                    num_children=1,
                    strength=float(schedule["noise_strength"]),
                )
                if not child_ids:
                    return
                self._forward_to(env, child_ids[0], 1.0, solver="euler", cfg=None)
                self._preview(env, child_ids[0], target_nfe, initial_budget)
            except (BudgetExceededError, InvalidActionError):
                return

    def _children_for_anchor(
        self,
        env: FlowTTSEnv,
        anchor: PreviewRecord,
        schedule: dict[str, float | int | bool | str],
    ) -> int:
        gap = self._gap_to_next(env, anchor.id)
        uncertainty = float(anchor.uncertainty or 0.0)
        max_children = max(1, int(schedule["backward_children"]))
        if gap <= 0.03 or uncertainty >= float(schedule["uncertainty_gate"]):
            return min(2, max_children)
        if gap <= 0.08:
            return 1
        return 1 if float(anchor.time) < 0.999 else 0

    def _safe_answer(self, env: FlowTTSEnv) -> AnswerRecord:
        try:
            return env.answer(rule="best_preview_score")
        except (BudgetExceededError, InvalidActionError):
            return env.answer(rule="latest_active")

    def _forward_to(
        self,
        env: FlowTTSEnv,
        particle_id: int,
        target_time: float,
        solver: str,
        cfg: dict[str, float] | None,
    ) -> None:
        state = env.get_state()
        particle = state.particles.get(particle_id)
        if particle is None:
            raise InvalidActionError(f"unknown particle_id: {particle_id}")
        if target_time <= float(particle.time):
            return
        env.forward(particle_id, target_time=target_time, solver=solver, cfg=cfg)

    def _preview(
        self,
        env: FlowTTSEnv,
        particle_id: int,
        target_nfe: int,
        initial_budget: int,
    ) -> PreviewRecord | None:
        if not self._can_afford(env, target_nfe, initial_budget, 1):
            return None
        return env.preview(particle_id, mode="clean_anchor", scorer="default")

    def _ranked_previews(self, env: FlowTTSEnv) -> list[PreviewRecord]:
        state = env.get_state()
        previews = [
            preview
            for preview in state.previews.values()
            if preview.score is not None
            and preview.particle_id in state.particles
            and state.particles[preview.particle_id].status != "pruned"
        ]
        return sorted(
            previews,
            key=lambda preview: (
                float(preview.score),
                -float(preview.uncertainty or 0.0),
                float(preview.time),
                -preview.id,
            ),
            reverse=True,
        )

    def _refinement_candidates(self, env: FlowTTSEnv) -> list[PreviewRecord]:
        previews = self._ranked_previews(env)
        if not previews:
            return []
        best = float(previews[0].score or 0.0)
        candidates: list[PreviewRecord] = []
        for preview in previews:
            gap = best - float(preview.score or 0.0)
            if gap <= 0.12 or not candidates:
                candidates.append(preview)
            if len(candidates) >= 3:
                break
        return candidates

    def _best_live_preview(self, env: FlowTTSEnv) -> PreviewRecord | None:
        previews = self._ranked_previews(env)
        return previews[0] if previews else None

    def _needs_confirmation(
        self,
        preview: PreviewRecord,
        schedule: dict[str, float | int | bool | str],
    ) -> bool:
        return (
            float(preview.time) < 0.999
            or float(preview.uncertainty or 0.0) > float(schedule["uncertainty_gate"])
        )

    def _gap_to_next(self, env: FlowTTSEnv, anchor_id: int) -> float:
        previews = self._ranked_previews(env)
        for index, preview in enumerate(previews):
            if preview.id != anchor_id:
                continue
            if index + 1 >= len(previews):
                return 1.0
            return abs(float(preview.score or 0.0) - float(previews[index + 1].score or 0.0))
        return 1.0

    def _child_cost(self, env: FlowTTSEnv, target_time: float) -> int:
        start_step = self._time_to_step_floor(env, target_time)
        return max(1, (len(env.time_grid) - 1) - start_step) + 1

    def _move_cost(self, env: FlowTTSEnv, particle_id: int, target_time: float) -> int:
        state = env.get_state()
        particle = state.particles.get(particle_id)
        if particle is None:
            return 0
        return max(0, self._time_to_step(env, target_time) - self._time_to_step(env, particle.time))

    def _steps_from_zero(self, env: FlowTTSEnv, target_time: float) -> int:
        return max(0, self._time_to_step(env, target_time))

    def _advance_time(self, env: FlowTTSEnv, particle_id: int, steps: int) -> float | None:
        state = env.get_state()
        particle = state.particles.get(particle_id)
        if particle is None:
            return None
        current_step = self._time_to_step(env, particle.time)
        target_step = min(len(env.time_grid) - 1, current_step + max(1, int(steps)))
        if target_step <= current_step:
            return None
        return float(env.time_grid[target_step])

    def _time_to_step(self, env: FlowTTSEnv, target_time: float) -> int:
        grid = list(env.time_grid)
        for index, value in enumerate(grid):
            if float(value) + 1e-9 >= float(target_time):
                return index
        return len(grid) - 1

    def _time_to_step_floor(self, env: FlowTTSEnv, target_time: float) -> int:
        grid = list(env.time_grid)
        step = 0
        for index, value in enumerate(grid):
            if float(value) - 1e-9 > float(target_time):
                break
            step = index
        return step

    def _is_active(self, env: FlowTTSEnv, particle_id: int) -> bool:
        state = env.get_state()
        particle = state.particles.get(particle_id)
        return particle is not None and particle.status == "active"

    def _can_afford(
        self,
        env: FlowTTSEnv,
        target_nfe: int,
        initial_budget: int,
        extra_cost: int,
    ) -> bool:
        extra_cost = max(0, int(extra_cost))
        return self._spent(env, initial_budget) + extra_cost <= target_nfe and env.budget_left >= extra_cost

    def _spent(self, env: FlowTTSEnv, initial_budget: int) -> int:
        return max(0, int(initial_budget - env.budget_left))
