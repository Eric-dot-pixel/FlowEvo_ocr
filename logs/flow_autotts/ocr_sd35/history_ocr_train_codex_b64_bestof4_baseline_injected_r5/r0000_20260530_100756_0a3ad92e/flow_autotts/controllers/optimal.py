"""Candidate controller for SD3.5 OCR discovery."""

from __future__ import annotations

from flow_autotts.core.env import FlowTTSEnv
from flow_autotts.core.errors import BudgetExceededError, InvalidActionError
from flow_autotts.core.state import AnswerRecord, PreviewRecord


class OptimalController:
    """Preview-driven controller with beta-scaled confirmation and refinement."""

    def solve(self, env: FlowTTSEnv, beta: float) -> AnswerRecord:
        beta = min(max(float(beta), 0.0), 1.0)
        schedule = self._schedule(env, beta)
        initial_budget = max(0, int(env.budget_left))
        target_nfe = min(initial_budget, int(schedule["target_nfe"]))

        try:
            root_ids = env.spawn(int(schedule["root_count"]))
        except InvalidActionError:
            return env.answer(rule="latest_active")

        if not root_ids:
            return env.answer(rule="latest_active")

        root_ids = root_ids[: self._affordable_roots(env, target_nfe, initial_budget, schedule)]
        if not root_ids:
            return env.answer(rule="latest_active")

        self._scout_roots(env, root_ids, schedule, target_nfe, initial_budget)
        self._prune_after_scout(env, schedule)
        self._finish_survivors(env, schedule, target_nfe, initial_budget)
        self._confirm_top_anchors(env, schedule, target_nfe, initial_budget)
        self._refine_best_anchors(env, schedule, target_nfe, initial_budget)
        self._spend_tail_budget(env, schedule, target_nfe, initial_budget)
        return self._safe_answer(env)

    def _schedule(self, env: FlowTTSEnv, beta: float) -> dict[str, float | int | bool | str]:
        step_count = max(1, len(env.time_grid) - 1)
        target_nfe = int(round(10.0 + 54.0 * (beta**1.1)))
        target_nfe = min(int(env.budget), max(10, target_nfe))

        if beta <= 0.0:
            return {
                "target_nfe": min(int(env.budget), 10),
                "root_count": 2,
                "scout_time": 0.6,
                "survivor_count": 1,
                "confirm_count": 1,
                "confirm_steps": 0,
                "backward_children": 0,
                "backward_time": 0.9,
                "noise_policy": "fresh_noise",
                "noise_strength": 1.0,
                "prune_margin": 0.35,
                "uncertainty_gate": 0.45,
                "late_preview_rounds": 0,
                "use_sde_scout": False,
                "step_count": step_count,
            }

        return {
            "target_nfe": target_nfe,
            "root_count": 2 + int(3 * beta),
            "scout_time": 0.4 + 0.2 * beta,
            "survivor_count": 1 + int(2 * beta),
            "confirm_count": 1 + int(2 * beta),
            "confirm_steps": int(1 + 2 * beta),
            "backward_children": int(1 + 3 * beta),
            "backward_time": 0.86 - 0.18 * beta,
            "noise_policy": "mixed_noise" if beta >= 0.65 else "fresh_noise",
            "noise_strength": 0.45 if beta >= 0.65 else 1.0,
            "prune_margin": 0.18 - 0.08 * beta,
            "uncertainty_gate": 0.52 - 0.18 * beta,
            "late_preview_rounds": int(1 + 3 * beta),
            "use_sde_scout": beta >= 0.75,
            "step_count": step_count,
        }

    def _affordable_roots(
        self,
        env: FlowTTSEnv,
        target_nfe: int,
        initial_budget: int,
        schedule: dict[str, float | int | bool | str],
    ) -> int:
        scout_step = self._time_to_step(env, float(schedule["scout_time"]))
        scout_cost = max(1, scout_step) + 1
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
        cfg = {"noise_scale": 0.01, "sde_type": "sde"} if solver == "sde" else None
        for particle_id in root_ids:
            if not self._can_afford(env, target_nfe, initial_budget, self._move_cost(env, particle_id, scout_time) + 1):
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
        if len(previews) <= int(schedule["survivor_count"]):
            return

        best = float(previews[0].score or 0.0)
        keep_ids: list[int] = []
        for index, preview in enumerate(previews):
            gap = best - float(preview.score or 0.0)
            if index < int(schedule["survivor_count"]) or gap <= float(schedule["prune_margin"]):
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
        if not previews:
            return
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
        confirm_count = int(schedule["confirm_count"])
        confirm_steps = int(schedule["confirm_steps"])
        if confirm_count <= 0 or confirm_steps <= 0:
            return

        previews = self._ranked_previews(env)[:confirm_count]
        for preview in previews:
            pid = preview.particle_id
            if not self._is_active(env, pid):
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
        children_budget = int(schedule["backward_children"])
        if children_budget <= 0:
            return

        launched = 0
        for anchor in self._refinement_candidates(env):
            if launched >= children_budget:
                break
            per_child_cost = self._child_cost(env, float(schedule["backward_time"]))
            if not self._can_afford(env, target_nfe, initial_budget, per_child_cost):
                return
            child_count = 1
            if launched + 1 < children_budget and self._gap_to_next(env, anchor.id) <= 0.05:
                child_count = min(2, children_budget - launched)
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
                if launched >= children_budget:
                    break
                if not self._can_afford(env, target_nfe, initial_budget, per_child_cost):
                    return
                try:
                    self._forward_to(env, child_id, 1.0, solver="euler", cfg=None)
                    self._preview(env, child_id, target_nfe, initial_budget)
                except (BudgetExceededError, InvalidActionError):
                    return
                launched += 1

    def _spend_tail_budget(
        self,
        env: FlowTTSEnv,
        schedule: dict[str, float | int | bool | str],
        target_nfe: int,
        initial_budget: int,
    ) -> None:
        rounds = int(schedule["late_preview_rounds"])
        while rounds > 0 and self._spent(env, initial_budget) < target_nfe:
            best = self._best_live_preview(env)
            if best is None:
                return
            pid = best.particle_id
            if self._is_active(env, pid):
                next_time = self._advance_time(env, pid, 1)
                if next_time is not None and self._can_afford(env, target_nfe, initial_budget, self._move_cost(env, pid, next_time) + 1):
                    try:
                        self._forward_to(env, pid, next_time, solver="euler", cfg=None)
                        self._preview(env, pid, target_nfe, initial_budget)
                        rounds -= 1
                        continue
                    except (BudgetExceededError, InvalidActionError):
                        return
            for anchor in self._refinement_candidates(env):
                child_cost = self._child_cost(env, float(schedule["backward_time"]))
                if not self._can_afford(env, target_nfe, initial_budget, child_cost):
                    break
                try:
                    child_ids = env.backward(
                        anchor.id,
                        target_time=float(schedule["backward_time"]),
                        noise_policy=str(schedule["noise_policy"]),
                        num_children=1,
                        strength=float(schedule["noise_strength"]),
                    )
                    for child_id in child_ids:
                        self._forward_to(env, child_id, 1.0, solver="euler", cfg=None)
                        self._preview(env, child_id, target_nfe, initial_budget)
                        rounds -= 1
                        break
                    if rounds <= 0:
                        return
                except (BudgetExceededError, InvalidActionError):
                    continue
            if self._can_afford(env, target_nfe, initial_budget, 1):
                try:
                    env.preview(pid, mode="clean_anchor", scorer="default")
                except (BudgetExceededError, InvalidActionError):
                    return
            return

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
        cfg: dict[str, float | str] | None,
    ) -> None:
        state = env.get_state()
        if particle_id not in state.particles:
            raise InvalidActionError(f"unknown particle_id: {particle_id}")
        current_time = state.particles[particle_id].time
        if target_time <= current_time:
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
            if gap <= 0.12 or len(candidates) == 0:
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
        return self._spent(env, initial_budget) + max(0, int(extra_cost)) <= target_nfe and env.budget_left >= extra_cost

    def _spent(self, env: FlowTTSEnv, initial_budget: int) -> int:
        return max(0, int(initial_budget - env.budget_left))
