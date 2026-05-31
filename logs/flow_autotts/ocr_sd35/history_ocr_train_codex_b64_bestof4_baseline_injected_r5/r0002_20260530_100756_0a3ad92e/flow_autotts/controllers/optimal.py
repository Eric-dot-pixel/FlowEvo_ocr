"""Candidate controller for SD3.5 OCR discovery."""

from __future__ import annotations

from flow_autotts.core.env import FlowTTSEnv
from flow_autotts.core.errors import BudgetExceededError, InvalidActionError
from flow_autotts.core.state import AnswerRecord, PreviewRecord


class OptimalController:
    """Preview-first controller with beta-scaled probe-before-commit refinement."""

    def solve(self, env: FlowTTSEnv, beta: float) -> AnswerRecord:
        beta = min(max(float(beta), 0.0), 1.0)
        schedule = self._schedule(env, beta)
        initial_budget = max(0, int(env.budget_left))
        target_nfe = min(initial_budget, int(schedule["target_nfe"]))

        try:
            root_id = env.spawn(1)[0]
        except (BudgetExceededError, InvalidActionError):
            return self._safe_answer(env)

        try:
            self._root_probe_ladder(env, root_id, schedule, target_nfe, initial_budget)
            self._refine_with_child_probes(env, schedule, target_nfe, initial_budget)
            self._spend_tail_budget(env, schedule, target_nfe, initial_budget)
        except BudgetExceededError:
            return self._safe_answer(env)

        return self._safe_answer(env)

    def _schedule(self, env: FlowTTSEnv, beta: float) -> dict[str, object]:
        target_map = {
            0.0: 10,
            0.25: 20,
            0.5: 36,
            0.75: 48,
            1.0: min(int(env.budget), 64),
        }
        beta_key = min(target_map, key=lambda key: abs(key - beta))
        target_nfe = min(int(env.budget), int(target_map[beta_key]))

        if beta <= 0.0:
            return {
                "target_nfe": target_nfe,
                "root_probe_times": (0.6, 0.8),
                "backward_time": 0.85,
                "promote_time": 0.9,
                "noise_policy": "inferred_noise",
                "noise_strength": 1.0,
                "max_child_probes": 0,
                "max_child_commits": 0,
                "max_children_per_backward": 1,
                "anchor_pool": 1,
                "anchor_gap": 0.0,
                "promote_gap": 0.0,
                "confirm_gap": 0.0,
                "prune_gap": 0.15,
                "uncertainty_gate": 0.45,
                "tail_rounds": 0,
                "tail_step_span": 1,
                "tail_reserve": 0,
            }
        if beta < 0.375:
            return {
                "target_nfe": target_nfe,
                "root_probe_times": (0.5, 0.8, 0.9),
                "backward_time": 0.8,
                "promote_time": 0.9,
                "noise_policy": "fresh_noise",
                "noise_strength": 1.0,
                "max_child_probes": 2,
                "max_child_commits": 1,
                "max_children_per_backward": 1,
                "anchor_pool": 2,
                "anchor_gap": 0.08,
                "promote_gap": 0.08,
                "confirm_gap": 0.06,
                "prune_gap": 0.12,
                "uncertainty_gate": 0.35,
                "tail_rounds": 1,
                "tail_step_span": 1,
                "tail_reserve": 1,
            }
        if beta < 0.625:
            return {
                "target_nfe": target_nfe,
                "root_probe_times": (0.5, 0.7, 0.9),
                "backward_time": 0.7,
                "promote_time": 0.9,
                "noise_policy": "fresh_noise",
                "noise_strength": 1.0,
                "max_child_probes": 4,
                "max_child_commits": 2,
                "max_children_per_backward": 1,
                "anchor_pool": 2,
                "anchor_gap": 0.1,
                "promote_gap": 0.08,
                "confirm_gap": 0.05,
                "prune_gap": 0.1,
                "uncertainty_gate": 0.28,
                "tail_rounds": 4,
                "tail_step_span": 1,
                "tail_reserve": 2,
            }
        if beta < 0.875:
            return {
                "target_nfe": target_nfe,
                "root_probe_times": (0.4, 0.6, 0.8, 0.9),
                "backward_time": 0.65,
                "promote_time": 1.0,
                "noise_policy": "mixed_noise",
                "noise_strength": 0.55,
                "max_child_probes": 6,
                "max_child_commits": 3,
                "max_children_per_backward": 2,
                "anchor_pool": 3,
                "anchor_gap": 0.12,
                "promote_gap": 0.08,
                "confirm_gap": 0.04,
                "prune_gap": 0.1,
                "uncertainty_gate": 0.22,
                "tail_rounds": 6,
                "tail_step_span": 1,
                "tail_reserve": 2,
            }
        return {
            "target_nfe": target_nfe,
            "root_probe_times": (0.4, 0.6, 0.8, 0.9),
            "backward_time": 0.6,
            "promote_time": 1.0,
            "noise_policy": "mixed_noise",
            "noise_strength": 0.4,
            "max_child_probes": 8,
            "max_child_commits": 4,
            "max_children_per_backward": 2,
            "anchor_pool": 4,
            "anchor_gap": 0.14,
            "promote_gap": 0.08,
            "confirm_gap": 0.04,
            "prune_gap": 0.1,
            "uncertainty_gate": 0.18,
            "tail_rounds": 10,
            "tail_step_span": 1,
            "tail_reserve": 0,
        }

    def _root_probe_ladder(
        self,
        env: FlowTTSEnv,
        particle_id: int,
        schedule: dict[str, object],
        target_nfe: int,
        initial_budget: int,
    ) -> None:
        for target_time in tuple(schedule["root_probe_times"]):
            cost = self._move_cost(env, particle_id, float(target_time)) + 1
            if not self._can_afford(env, target_nfe, initial_budget, cost):
                return
            try:
                self._forward_to(env, particle_id, float(target_time))
                self._preview(env, particle_id, target_nfe, initial_budget)
            except (BudgetExceededError, InvalidActionError):
                return

    def _refine_with_child_probes(
        self,
        env: FlowTTSEnv,
        schedule: dict[str, object],
        target_nfe: int,
        initial_budget: int,
    ) -> None:
        max_child_probes = int(schedule["max_child_probes"])
        max_child_commits = int(schedule["max_child_commits"])
        if max_child_probes <= 0:
            return

        launched_probes = 0
        committed_children = 0
        anchor_uses: dict[int, int] = {}

        while launched_probes < max_child_probes:
            anchor = self._select_probe_anchor(env, schedule, anchor_uses)
            if anchor is None:
                return
            if not self._can_afford(env, target_nfe, initial_budget, 1):
                return

            child_batch = 1
            if (
                int(schedule["max_children_per_backward"]) > 1
                and launched_probes + 1 < max_child_probes
                and self._top_gap(env) <= float(schedule["confirm_gap"])
                and self._remaining_to_target(env, target_nfe, initial_budget) >= 2
            ):
                child_batch = 2

            try:
                child_ids = env.backward(
                    anchor.id,
                    target_time=float(schedule["backward_time"]),
                    noise_policy=str(schedule["noise_policy"]),
                    num_children=child_batch,
                    strength=float(schedule["noise_strength"]),
                )
            except (BudgetExceededError, InvalidActionError):
                anchor_uses[anchor.id] = anchor_uses.get(anchor.id, 0) + 1
                continue

            anchor_uses[anchor.id] = anchor_uses.get(anchor.id, 0) + len(child_ids)

            for child_id in child_ids:
                if launched_probes >= max_child_probes:
                    return
                child_preview = self._preview(env, child_id, target_nfe, initial_budget)
                launched_probes += 1
                if child_preview is None:
                    return

                if committed_children >= max_child_commits:
                    self._prune_if_active(env, child_id)
                    continue

                if not self._should_promote_child(env, anchor, child_preview, schedule):
                    self._prune_if_active(env, child_id)
                    continue

                promote_cost = self._move_cost(env, child_id, float(schedule["promote_time"])) + 1
                if not self._can_afford(env, target_nfe, initial_budget, promote_cost):
                    continue

                try:
                    self._forward_to(env, child_id, float(schedule["promote_time"]))
                    self._preview(env, child_id, target_nfe, initial_budget)
                    committed_children += 1
                    self._prune_clear_losers(env, schedule)
                except (BudgetExceededError, InvalidActionError):
                    return

    def _spend_tail_budget(
        self,
        env: FlowTTSEnv,
        schedule: dict[str, object],
        target_nfe: int,
        initial_budget: int,
    ) -> None:
        rounds = int(schedule["tail_rounds"])
        while rounds > 0 and self._spent(env, initial_budget) < target_nfe:
            if self._remaining_to_target(env, target_nfe, initial_budget) <= int(schedule["tail_reserve"]):
                return

            candidate = self._select_tail_candidate(env, schedule)
            if candidate is not None:
                next_time = self._next_time(env, candidate.particle_id, int(schedule["tail_step_span"]))
                if next_time is not None:
                    cost = self._move_cost(env, candidate.particle_id, next_time) + 1
                    if self._can_afford(env, target_nfe, initial_budget, cost):
                        try:
                            self._forward_to(env, candidate.particle_id, next_time)
                            self._preview(env, candidate.particle_id, target_nfe, initial_budget)
                            rounds -= 1
                            continue
                        except (BudgetExceededError, InvalidActionError):
                            return

            if not self._open_tail_probe(env, schedule, target_nfe, initial_budget):
                return
            rounds -= 1

    def _open_tail_probe(
        self,
        env: FlowTTSEnv,
        schedule: dict[str, object],
        target_nfe: int,
        initial_budget: int,
    ) -> bool:
        anchor = self._select_probe_anchor(env, schedule, {})
        if anchor is None or not self._can_afford(env, target_nfe, initial_budget, 1):
            return False
        try:
            child_id = env.backward(
                anchor.id,
                target_time=float(schedule["backward_time"]),
                noise_policy=str(schedule["noise_policy"]),
                num_children=1,
                strength=float(schedule["noise_strength"]),
            )[0]
            child_preview = self._preview(env, child_id, target_nfe, initial_budget)
            if child_preview is None:
                return False
            if self._should_promote_child(env, anchor, child_preview, schedule):
                promote_cost = self._move_cost(env, child_id, float(schedule["promote_time"])) + 1
                if self._can_afford(env, target_nfe, initial_budget, promote_cost):
                    self._forward_to(env, child_id, float(schedule["promote_time"]))
                    self._preview(env, child_id, target_nfe, initial_budget)
                else:
                    self._prune_if_active(env, child_id)
            else:
                self._prune_if_active(env, child_id)
            return True
        except (BudgetExceededError, InvalidActionError):
            return False

    def _select_probe_anchor(
        self,
        env: FlowTTSEnv,
        schedule: dict[str, object],
        anchor_uses: dict[int, int],
    ) -> PreviewRecord | None:
        ranked = self._ranked_previews(env)
        if not ranked:
            return None

        best_score = float(ranked[0].score or 0.0)
        pool: list[PreviewRecord] = []
        for preview in ranked:
            gap = best_score - float(preview.score or 0.0)
            if len(pool) < int(schedule["anchor_pool"]) or gap <= float(schedule["anchor_gap"]):
                pool.append(preview)
            if len(pool) >= int(schedule["anchor_pool"]) and gap > float(schedule["anchor_gap"]):
                break

        if not pool:
            return ranked[0]

        return min(
            pool,
            key=lambda preview: (
                anchor_uses.get(preview.id, 0),
                -(float(preview.score or 0.0)),
                -(float(preview.time)),
                preview.id,
            ),
        )

    def _select_tail_candidate(
        self,
        env: FlowTTSEnv,
        schedule: dict[str, object],
    ) -> PreviewRecord | None:
        ranked = self._ranked_previews(env)
        if not ranked:
            return None

        if len(ranked) >= 2:
            gap = float(ranked[0].score or 0.0) - float(ranked[1].score or 0.0)
            if gap <= float(schedule["confirm_gap"]) and self._is_active(env, ranked[1].particle_id):
                return ranked[1]

        best = ranked[0]
        if not self._is_active(env, best.particle_id):
            return None

        return best

    def _should_promote_child(
        self,
        env: FlowTTSEnv,
        anchor: PreviewRecord,
        child_preview: PreviewRecord,
        schedule: dict[str, object],
    ) -> bool:
        ranked = self._ranked_previews(env)
        best_score = float(ranked[0].score or 0.0) if ranked else float(child_preview.score or 0.0)
        child_score = float(child_preview.score or 0.0)
        anchor_score = float(anchor.score or 0.0)
        uncertainty_gain = float(anchor.uncertainty or 0.0) - float(child_preview.uncertainty or 0.0)
        drift_gain = float(anchor.drift or 0.0) - float(child_preview.drift or 0.0)
        return (
            child_score >= best_score - float(schedule["promote_gap"])
            or child_score >= anchor_score
            or uncertainty_gain >= 0.08
            or drift_gain >= 0.02
        )

    def _prune_clear_losers(
        self,
        env: FlowTTSEnv,
        schedule: dict[str, object],
    ) -> None:
        ranked = self._ranked_previews(env)
        if len(ranked) < 2:
            return

        best_score = float(ranked[0].score or 0.0)
        prune_ids: list[int] = []
        for preview in ranked[1:]:
            if not self._is_active(env, preview.particle_id):
                continue
            if best_score - float(preview.score or 0.0) > float(schedule["prune_gap"]):
                prune_ids.append(preview.particle_id)
        if not prune_ids:
            return
        try:
            env.prune(sorted(set(prune_ids)))
        except InvalidActionError:
            return

    def _forward_to(self, env: FlowTTSEnv, particle_id: int, target_time: float) -> None:
        state = env.get_state()
        particle = state.particles.get(particle_id)
        if particle is None:
            raise InvalidActionError(f"unknown particle_id: {particle_id}")
        if target_time <= float(particle.time):
            return
        env.forward(particle_id, target_time=target_time, solver="euler")

    def _preview(
        self,
        env: FlowTTSEnv,
        particle_id: int,
        target_nfe: int,
        initial_budget: int,
    ) -> PreviewRecord | None:
        if not self._can_afford(env, target_nfe, initial_budget, 1):
            return None
        try:
            return env.preview(particle_id, mode="clean_anchor", scorer="default")
        except (BudgetExceededError, InvalidActionError):
            return None

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
                float(preview.score or 0.0),
                -float(preview.uncertainty or 0.0),
                float(preview.time),
                -preview.id,
            ),
            reverse=True,
        )

    def _move_cost(self, env: FlowTTSEnv, particle_id: int, target_time: float) -> int:
        state = env.get_state()
        particle = state.particles.get(particle_id)
        if particle is None:
            return 0
        current_step = self._time_to_step(env, float(particle.time))
        target_step = self._time_to_step(env, target_time)
        return max(0, target_step - current_step)

    def _time_to_step(self, env: FlowTTSEnv, target_time: float) -> int:
        for index, grid_time in enumerate(env.time_grid):
            if float(grid_time) + 1e-9 >= float(target_time):
                return index
        return len(env.time_grid) - 1

    def _next_time(self, env: FlowTTSEnv, particle_id: int, step_span: int) -> float | None:
        state = env.get_state()
        particle = state.particles.get(particle_id)
        if particle is None or particle.status != "active":
            return None
        current_step = self._time_to_step(env, float(particle.time))
        target_step = min(len(env.time_grid) - 1, current_step + max(1, int(step_span)))
        if target_step <= current_step:
            return None
        return float(env.time_grid[target_step])

    def _top_gap(self, env: FlowTTSEnv) -> float:
        ranked = self._ranked_previews(env)
        if len(ranked) < 2:
            return 1.0
        return abs(float(ranked[0].score or 0.0) - float(ranked[1].score or 0.0))

    def _prune_if_active(self, env: FlowTTSEnv, particle_id: int) -> None:
        if not self._is_active(env, particle_id):
            return
        try:
            env.prune([particle_id])
        except InvalidActionError:
            return

    def _is_active(self, env: FlowTTSEnv, particle_id: int) -> bool:
        state = env.get_state()
        particle = state.particles.get(particle_id)
        return particle is not None and particle.status == "active"

    def _remaining_to_target(self, env: FlowTTSEnv, target_nfe: int, initial_budget: int) -> int:
        return max(0, int(target_nfe) - self._spent(env, initial_budget))

    def _can_afford(
        self,
        env: FlowTTSEnv,
        target_nfe: int,
        initial_budget: int,
        extra_cost: int,
    ) -> bool:
        cost = max(0, int(extra_cost))
        return self._spent(env, initial_budget) + cost <= int(target_nfe) and env.budget_left >= cost

    def _spent(self, env: FlowTTSEnv, initial_budget: int) -> int:
        return max(0, int(initial_budget - env.budget_left))

    def _safe_answer(self, env: FlowTTSEnv) -> AnswerRecord:
        try:
            return env.answer(rule="best_preview_score")
        except (BudgetExceededError, InvalidActionError):
            return env.answer(rule="latest_active")
