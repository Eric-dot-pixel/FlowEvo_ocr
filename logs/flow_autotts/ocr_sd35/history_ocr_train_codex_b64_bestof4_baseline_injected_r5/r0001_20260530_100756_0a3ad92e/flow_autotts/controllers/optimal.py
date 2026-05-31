"""Candidate controller for SD3.5 OCR discovery."""

from __future__ import annotations

from flow_autotts.core.env import FlowTTSEnv
from flow_autotts.core.errors import BudgetExceededError, InvalidActionError
from flow_autotts.core.state import AnswerRecord, PreviewRecord


class OptimalController:
    """Preview-driven controller with beta-scaled width, confirmation, and refinement."""

    def solve(self, env: FlowTTSEnv, beta: float) -> AnswerRecord:
        beta = min(max(float(beta), 0.0), 1.0)
        schedule = self._schedule(env, beta)
        initial_budget = max(0, int(env.budget_left))
        target_nfe = min(initial_budget, int(schedule["target_nfe"]))

        try:
            root_ids = env.spawn(1)
        except InvalidActionError:
            return self._safe_answer(env)
        if not root_ids:
            return self._safe_answer(env)

        root_id = root_ids[0]
        try:
            self._forward_to(env, root_id, float(schedule["scout_time"]), solver="euler", cfg=None)
            self._preview(env, root_id, target_nfe, initial_budget)
        except (BudgetExceededError, InvalidActionError):
            return self._safe_answer(env)

        if beta <= 0.0:
            self._finish_particle(env, root_id, target_nfe, initial_budget, preview_after=True)
            self._tail_confirm(env, schedule, target_nfe, initial_budget)
            return self._safe_answer(env)

        if bool(schedule["allow_second_root"]):
            self._maybe_launch_second_root(env, schedule, target_nfe, initial_budget)

        self._prune_after_scout(env, schedule)
        self._finish_ranked_roots(env, schedule, target_nfe, initial_budget)
        self._confirm_ranked_previews(env, schedule, target_nfe, initial_budget)
        self._refine_top_anchors(env, schedule, target_nfe, initial_budget)
        self._spend_tail_budget(env, schedule, target_nfe, initial_budget)
        return self._safe_answer(env)

    def _schedule(self, env: FlowTTSEnv, beta: float) -> dict[str, float | int | bool | str]:
        target_map = {
            0.0: 10,
            0.25: 20,
            0.5: 36,
            0.75: 48,
            1.0: 64,
        }
        beta_key = min(target_map, key=lambda key: abs(key - beta))
        target_nfe = min(int(env.budget), int(target_map[beta_key]))

        if beta <= 0.0:
            return {
                "target_nfe": min(int(env.budget), 10),
                "scout_time": 0.6,
                "second_root_time": 0.0,
                "allow_second_root": False,
                "survivor_count": 1,
                "confirm_rounds": 1,
                "confirm_gap": 0.0,
                "uncertainty_gate": 0.3,
                "backward_children": 0,
                "backward_time": 0.88,
                "noise_policy": "fresh_noise",
                "noise_strength": 1.0,
                "prune_margin": 0.25,
                "tail_rounds": 1,
            }

        return {
            "target_nfe": target_nfe,
            "scout_time": 0.35 + 0.15 * beta,
            "second_root_time": 0.3 + 0.1 * beta,
            "allow_second_root": beta >= 0.25,
            "survivor_count": 1 if beta < 0.55 else 2,
            "confirm_rounds": 1 + int(beta >= 0.5) + int(beta >= 0.85),
            "confirm_gap": 0.07 - 0.02 * beta,
            "uncertainty_gate": 0.48 - 0.14 * beta,
            "backward_children": int(beta >= 0.25) + int(beta >= 0.6) + int(beta >= 0.9),
            "backward_time": 0.86 - 0.18 * beta,
            "noise_policy": "mixed_noise" if beta >= 0.75 else "fresh_noise",
            "noise_strength": 0.45 if beta >= 0.75 else 1.0,
            "prune_margin": 0.16 - 0.05 * beta,
            "tail_rounds": 1 + int(beta >= 0.5) + int(beta >= 0.75),
        }

    def _maybe_launch_second_root(
        self,
        env: FlowTTSEnv,
        schedule: dict[str, float | int | bool | str],
        target_nfe: int,
        initial_budget: int,
    ) -> None:
        previews = self._ranked_previews(env)
        if not previews:
            return
        best = previews[0]
        should_expand = (
            float(best.score or 0.0) < 0.7
            or float(best.uncertainty or 0.0) > float(schedule["uncertainty_gate"])
        )
        if not should_expand:
            return

        second_time = float(schedule["second_root_time"])
        full_time = 1.0
        cost = self._spawned_particle_cost(env, 0.0, second_time) + 1
        if beta_like_high(schedule):
            cost += self._spawned_particle_cost(env, second_time, full_time)
        if not self._can_afford(env, target_nfe, initial_budget, cost):
            return

        try:
            child_root = env.spawn(1)[0]
            self._forward_to(env, child_root, second_time, solver="euler", cfg=None)
            self._preview(env, child_root, target_nfe, initial_budget)
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

    def _finish_ranked_roots(
        self,
        env: FlowTTSEnv,
        schedule: dict[str, float | int | bool | str],
        target_nfe: int,
        initial_budget: int,
    ) -> None:
        previews = self._ranked_previews(env)
        if not previews:
            return
        for preview in previews[: int(schedule["survivor_count"])]:
            self._finish_particle(env, preview.particle_id, target_nfe, initial_budget, preview_after=True)

    def _confirm_ranked_previews(
        self,
        env: FlowTTSEnv,
        schedule: dict[str, float | int | bool | str],
        target_nfe: int,
        initial_budget: int,
    ) -> None:
        rounds = int(schedule["confirm_rounds"])
        while rounds > 0:
            previews = self._ranked_previews(env)
            if len(previews) < 2:
                candidate = previews[0] if previews else None
            else:
                gap = float(previews[0].score or 0.0) - float(previews[1].score or 0.0)
                candidate = previews[1] if gap <= float(schedule["confirm_gap"]) else previews[0]
            if candidate is None:
                return
            if not self._needs_confirmation(candidate, schedule):
                return
            next_time = self._next_time(env, candidate.particle_id)
            if next_time is None:
                if self._can_afford(env, target_nfe, initial_budget, 1):
                    try:
                        env.preview(candidate.particle_id, mode="clean_anchor", scorer="default")
                    except (BudgetExceededError, InvalidActionError):
                        return
                return
            cost = self._move_cost(env, candidate.particle_id, next_time) + 1
            if not self._can_afford(env, target_nfe, initial_budget, cost):
                return
            try:
                self._forward_to(env, candidate.particle_id, next_time, solver="euler", cfg=None)
                self._preview(env, candidate.particle_id, target_nfe, initial_budget)
            except (BudgetExceededError, InvalidActionError):
                return
            rounds -= 1

    def _refine_top_anchors(
        self,
        env: FlowTTSEnv,
        schedule: dict[str, float | int | bool | str],
        target_nfe: int,
        initial_budget: int,
    ) -> None:
        max_children = int(schedule["backward_children"])
        if max_children <= 0:
            return
        launched = 0
        for anchor in self._refinement_candidates(env, schedule):
            if launched >= max_children:
                return
            cost = self._child_cost(env, float(schedule["backward_time"]))
            if not self._can_afford(env, target_nfe, initial_budget, cost):
                return
            try:
                child_ids = env.backward(
                    anchor.id,
                    target_time=float(schedule["backward_time"]),
                    noise_policy=str(schedule["noise_policy"]),
                    num_children=1,
                    strength=float(schedule["noise_strength"]),
                )
            except (BudgetExceededError, InvalidActionError):
                continue
            for child_id in child_ids:
                if launched >= max_children:
                    return
                self._finish_particle(env, child_id, target_nfe, initial_budget, preview_after=True)
                launched += 1

    def _spend_tail_budget(
        self,
        env: FlowTTSEnv,
        schedule: dict[str, float | int | bool | str],
        target_nfe: int,
        initial_budget: int,
    ) -> None:
        rounds = int(schedule["tail_rounds"])
        while rounds > 0 and self._spent(env, initial_budget) < target_nfe:
            if self._maybe_open_tail_root(env, schedule, target_nfe, initial_budget):
                rounds -= 1
                continue
            previews = self._ranked_previews(env)
            if not previews:
                return
            gap = self._gap_between_top(previews)
            best = previews[0]
            if gap <= float(schedule["confirm_gap"]) and int(schedule["backward_children"]) > 0:
                cost = self._child_cost(env, float(schedule["backward_time"]))
                if self._can_afford(env, target_nfe, initial_budget, cost):
                    try:
                        child_id = env.backward(
                            best.id,
                            target_time=float(schedule["backward_time"]),
                            noise_policy=str(schedule["noise_policy"]),
                            num_children=1,
                            strength=float(schedule["noise_strength"]),
                        )[0]
                        self._finish_particle(env, child_id, target_nfe, initial_budget, preview_after=True)
                        rounds -= 1
                        continue
                    except (BudgetExceededError, InvalidActionError):
                        return
            if self._needs_confirmation(best, schedule):
                next_time = self._next_time(env, best.particle_id)
                if next_time is not None:
                    cost = self._move_cost(env, best.particle_id, next_time) + 1
                    if self._can_afford(env, target_nfe, initial_budget, cost):
                        try:
                            self._forward_to(env, best.particle_id, next_time, solver="euler", cfg=None)
                            self._preview(env, best.particle_id, target_nfe, initial_budget)
                            rounds -= 1
                            continue
                        except (BudgetExceededError, InvalidActionError):
                            return
            self._tail_confirm(env, schedule, target_nfe, initial_budget)
            return

    def _maybe_open_tail_root(
        self,
        env: FlowTTSEnv,
        schedule: dict[str, float | int | bool | str],
        target_nfe: int,
        initial_budget: int,
    ) -> bool:
        remaining = target_nfe - self._spent(env, initial_budget)
        finish_cost = self._spawned_particle_cost(env, 0.0, 1.0) + 1
        if remaining < finish_cost:
            return False
        if int(schedule["target_nfe"]) < 36:
            return False
        try:
            pid = env.spawn(1)[0]
            self._finish_particle(env, pid, target_nfe, initial_budget, preview_after=True)
            return True
        except (BudgetExceededError, InvalidActionError):
            return False

    def _tail_confirm(
        self,
        env: FlowTTSEnv,
        schedule: dict[str, float | int | bool | str],
        target_nfe: int,
        initial_budget: int,
    ) -> None:
        best = self._best_live_preview(env)
        if best is None:
            return
        if self._can_afford(env, target_nfe, initial_budget, 1):
            try:
                env.preview(best.particle_id, mode="clean_anchor", scorer="default")
            except (BudgetExceededError, InvalidActionError):
                return

    def _finish_particle(
        self,
        env: FlowTTSEnv,
        particle_id: int,
        target_nfe: int,
        initial_budget: int,
        preview_after: bool,
    ) -> None:
        if not self._is_active(env, particle_id):
            return
        cost = self._move_cost(env, particle_id, 1.0) + (1 if preview_after else 0)
        if not self._can_afford(env, target_nfe, initial_budget, cost):
            return
        try:
            self._forward_to(env, particle_id, 1.0, solver="euler", cfg=None)
            if preview_after:
                self._preview(env, particle_id, target_nfe, initial_budget)
        except (BudgetExceededError, InvalidActionError):
            return

    def _safe_answer(self, env: FlowTTSEnv) -> AnswerRecord:
        try:
            return env.answer(rule="best_preview_score")
        except (BudgetExceededError, InvalidActionError):
            return env.answer(rule="latest_active")

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

    def _forward_to(
        self,
        env: FlowTTSEnv,
        particle_id: int,
        target_time: float,
        solver: str,
        cfg: dict[str, float | str] | None,
    ) -> None:
        state = env.get_state()
        particle = state.particles.get(particle_id)
        if particle is None or particle.status != "active":
            raise InvalidActionError(f"invalid particle_id: {particle_id}")
        if target_time <= float(particle.time):
            return
        env.forward(particle_id, target_time=target_time, solver=solver, cfg=cfg)

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

    def _refinement_candidates(
        self,
        env: FlowTTSEnv,
        schedule: dict[str, float | int | bool | str],
    ) -> list[PreviewRecord]:
        previews = self._ranked_previews(env)
        if not previews:
            return []
        best = float(previews[0].score or 0.0)
        candidates: list[PreviewRecord] = []
        for preview in previews[:3]:
            gap = best - float(preview.score or 0.0)
            uncertain = float(preview.uncertainty or 0.0) >= float(schedule["uncertainty_gate"])
            if gap <= max(0.12, float(schedule["confirm_gap"]) + 0.03) or uncertain or not candidates:
                candidates.append(preview)
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

    def _gap_between_top(self, previews: list[PreviewRecord]) -> float:
        if len(previews) < 2:
            return 1.0
        return abs(float(previews[0].score or 0.0) - float(previews[1].score or 0.0))

    def _child_cost(self, env: FlowTTSEnv, target_time: float) -> int:
        start_step = self._time_to_step_floor(env, target_time)
        return max(1, (len(env.time_grid) - 1) - start_step) + 1

    def _spawned_particle_cost(self, env: FlowTTSEnv, start_time: float, target_time: float) -> int:
        return max(0, self._time_to_step(env, target_time) - self._time_to_step_floor(env, start_time))

    def _move_cost(self, env: FlowTTSEnv, particle_id: int, target_time: float) -> int:
        state = env.get_state()
        particle = state.particles.get(particle_id)
        if particle is None:
            return 0
        return max(0, self._time_to_step(env, target_time) - self._time_to_step(env, particle.time))

    def _next_time(self, env: FlowTTSEnv, particle_id: int) -> float | None:
        state = env.get_state()
        particle = state.particles.get(particle_id)
        if particle is None:
            return None
        current_step = self._time_to_step(env, particle.time)
        if current_step >= len(env.time_grid) - 1:
            return None
        return float(env.time_grid[current_step + 1])

    def _time_to_step(self, env: FlowTTSEnv, target_time: float) -> int:
        for index, value in enumerate(env.time_grid):
            if float(value) + 1e-9 >= float(target_time):
                return index
        return len(env.time_grid) - 1

    def _time_to_step_floor(self, env: FlowTTSEnv, target_time: float) -> int:
        step = 0
        for index, value in enumerate(env.time_grid):
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


def beta_like_high(schedule: dict[str, float | int | bool | str]) -> bool:
    return int(schedule["target_nfe"]) >= 48
