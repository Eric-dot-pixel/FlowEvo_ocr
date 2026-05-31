"""Candidate controller for SD3.5 OCR discovery."""

from __future__ import annotations

from flow_autotts.core.env import FlowTTSEnv
from flow_autotts.core.errors import BudgetExceededError, InvalidActionError
from flow_autotts.core.state import AnswerRecord, PreviewRecord


class OptimalController:
    """Preview-driven controller with bounded width and probe-before-commit refinement."""

    def solve(self, env: FlowTTSEnv, beta: float) -> AnswerRecord:
        beta = min(max(float(beta), 0.0), 1.0)
        initial_budget = max(0, int(env.budget_left))
        schedule = self._schedule(env, beta)
        target_nfe = min(initial_budget, int(schedule["target_nfe"]))

        if target_nfe <= 0:
            return self._safe_answer(env)
        if beta <= 0.0:
            return self._solve_locked_beta_zero(env, target_nfe, initial_budget)

        try:
            root_ids = env.spawn(int(schedule["root_count"]))
        except InvalidActionError:
            return self._safe_answer(env)
        if not root_ids:
            return self._safe_answer(env)

        root_ids = root_ids[: self._affordable_roots(env, schedule, target_nfe, initial_budget)]
        if not root_ids:
            return self._safe_answer(env)

        self._scout_roots(env, root_ids, schedule, target_nfe, initial_budget)
        self._prune_losers(env, schedule)
        self._advance_contenders(env, int(schedule["mid_step"]), schedule, target_nfe, initial_budget)
        self._prune_losers(env, schedule)
        self._advance_contenders(env, int(schedule["late_step"]), schedule, target_nfe, initial_budget)
        self._prune_losers(env, schedule)
        self._probe_refinements(env, schedule, target_nfe, initial_budget)
        self._finish_best_active(env, int(schedule["finish_budget"]), target_nfe, initial_budget)
        self._tail_actions(env, schedule, target_nfe, initial_budget)
        return self._safe_answer(env)

    def _solve_locked_beta_zero(
        self,
        env: FlowTTSEnv,
        target_nfe: int,
        initial_budget: int,
    ) -> AnswerRecord:
        try:
            particle_id = env.spawn(1)[0]
        except (BudgetExceededError, InvalidActionError):
            return self._safe_answer(env)

        num_steps = max(1, len(env.time_grid) - 1)
        late_step = max(1, min(num_steps - 2, target_nfe - 4))
        final_probe_step = max(late_step + 1, min(num_steps - 1, target_nfe - 2))

        for target_step in [late_step, final_probe_step]:
            cost = self._move_cost_by_step(env, particle_id, target_step) + 1
            if not self._can_afford(env, target_nfe, initial_budget, cost):
                break
            try:
                self._forward_to_step(env, particle_id, target_step, solver="euler", cfg=None)
                self._preview(env, particle_id, target_nfe, initial_budget)
            except (BudgetExceededError, InvalidActionError):
                break
        return self._safe_answer(env)

    def _schedule(self, env: FlowTTSEnv, beta: float) -> dict[str, float | int | bool | str]:
        num_steps = max(1, len(env.time_grid) - 1)
        target_nfe = min(int(env.budget), self._beta_target_nfe(beta))

        if beta <= 0.0:
            return {
                "target_nfe": min(int(env.budget), 10),
                "root_count": 1,
                "scout_step": max(1, num_steps // 2),
                "mid_step": max(1, num_steps - 3),
                "late_step": max(1, num_steps - 2),
                "survivor_count": 1,
                "prune_margin": 0.30,
                "hold_margin": 0.12,
                "uncertainty_gate": 0.50,
                "plateau_gate": 0.03,
                "branch_step": max(1, num_steps - 3),
                "late_probe_step": max(1, num_steps - 2),
                "max_child_probes": 0,
                "finish_budget": 0,
                "tail_rounds": 0,
                "noise_policy": "fresh_noise",
                "noise_strength": 1.0,
                "use_sde_scout": False,
                "split_gap": 0.05,
            }

        scout_step = min(num_steps - 2, 3 + int(beta >= 0.5))
        mid_step = min(num_steps - 1, scout_step + 3)
        late_step = min(num_steps - 1, mid_step + 2)

        return {
            "target_nfe": target_nfe,
            "root_count": 1 + int(beta >= 0.5) + int(beta >= 0.75) + int(beta >= 0.95),
            "scout_step": scout_step,
            "mid_step": mid_step,
            "late_step": late_step,
            "survivor_count": 1 + int(beta >= 0.75),
            "prune_margin": 0.18 - 0.05 * beta,
            "hold_margin": 0.08 - 0.03 * beta,
            "uncertainty_gate": 0.42 - 0.10 * beta,
            "plateau_gate": 0.022 - 0.006 * beta,
            "branch_step": max(scout_step + 2, late_step - 3),
            "late_probe_step": late_step,
            "max_child_probes": 1
            + int(beta >= 0.25)
            + int(beta >= 0.5)
            + int(beta >= 0.75)
            + int(beta >= 0.95),
            "finish_budget": 1 + int(beta >= 0.75),
            "tail_rounds": 1 + int(beta >= 0.5) + int(beta >= 0.75) + int(beta >= 0.95),
            "noise_policy": "mixed_noise" if beta >= 0.75 else "fresh_noise",
            "noise_strength": 0.45 if beta >= 0.75 else 1.0,
            "use_sde_scout": beta >= 0.95,
            "split_gap": 0.05 - 0.02 * beta,
        }

    def _beta_target_nfe(self, beta: float) -> int:
        knots = [
            (0.0, 10.0),
            (0.25, 20.0),
            (0.5, 36.0),
            (0.75, 48.0),
            (1.0, 64.0),
        ]
        beta = min(max(float(beta), 0.0), 1.0)
        for index in range(1, len(knots)):
            left_beta, left_nfe = knots[index - 1]
            right_beta, right_nfe = knots[index]
            if beta <= right_beta:
                span = max(1e-9, right_beta - left_beta)
                mix = (beta - left_beta) / span
                return int(round(left_nfe + mix * (right_nfe - left_nfe)))
        return int(knots[-1][1])

    def _affordable_roots(
        self,
        env: FlowTTSEnv,
        schedule: dict[str, float | int | bool | str],
        target_nfe: int,
        initial_budget: int,
    ) -> int:
        scout_step = int(schedule["scout_step"])
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
        scout_step = int(schedule["scout_step"])
        for index, particle_id in enumerate(root_ids):
            solver = "euler"
            cfg: dict[str, float] | None = None
            if bool(schedule["use_sde_scout"]) and index == len(root_ids) - 1 and len(root_ids) > 1:
                solver = "sde"
                cfg = {"noise_scale": 0.008, "sigma_max": 1.15, "min_time": 0.02}
            cost = self._move_cost_by_step(env, particle_id, scout_step) + 1
            if not self._can_afford(env, target_nfe, initial_budget, cost):
                return
            try:
                self._forward_to_step(env, particle_id, scout_step, solver=solver, cfg=cfg)
                self._preview(env, particle_id, target_nfe, initial_budget)
            except (BudgetExceededError, InvalidActionError):
                return

    def _advance_contenders(
        self,
        env: FlowTTSEnv,
        target_step: int,
        schedule: dict[str, float | int | bool | str],
        target_nfe: int,
        initial_budget: int,
    ) -> None:
        previews = self._contender_previews(env, schedule)
        for preview in previews:
            particle = self._particle(env, preview.particle_id)
            if particle is None or particle.status != "active":
                continue
            current_step = self._time_to_step(env, particle.time)
            if target_step <= current_step:
                continue
            cost = self._move_cost_by_step(env, preview.particle_id, target_step) + 1
            if not self._can_afford(env, target_nfe, initial_budget, cost):
                continue
            try:
                self._forward_to_step(env, preview.particle_id, target_step, solver="euler", cfg=None)
                self._preview(env, preview.particle_id, target_nfe, initial_budget)
            except (BudgetExceededError, InvalidActionError):
                continue

    def _probe_refinements(
        self,
        env: FlowTTSEnv,
        schedule: dict[str, float | int | bool | str],
        target_nfe: int,
        initial_budget: int,
    ) -> None:
        launched = 0
        max_children = int(schedule["max_child_probes"])
        probe_step = int(schedule["late_probe_step"])
        branch_step = int(schedule["branch_step"])

        while launched < max_children and self._spent(env, initial_budget) < target_nfe:
            anchors = self._branch_candidates(env, schedule)
            if not anchors:
                return
            remaining = max_children - launched
            made_progress = False
            for anchor in anchors:
                if remaining <= 0:
                    break
                if not self._should_probe_anchor(env, anchor, schedule, target_nfe, initial_budget):
                    continue
                child_quota = 1
                if len(anchors) == 1 and remaining >= 2 and self._anchor_needs_more_exploration(env, anchor, schedule):
                    child_quota = 2
                child_quota = min(child_quota, remaining)
                per_child_cost = self._child_probe_cost(env, branch_step, probe_step)
                if not self._can_afford(env, target_nfe, initial_budget, child_quota * per_child_cost):
                    child_quota = max(
                        0,
                        (target_nfe - self._spent(env, initial_budget)) // max(1, per_child_cost),
                    )
                if child_quota <= 0:
                    continue
                try:
                    child_ids = env.backward(
                        anchor.id,
                        target_time=float(env.time_grid[branch_step]),
                        noise_policy=str(schedule["noise_policy"]),
                        num_children=child_quota,
                        strength=float(schedule["noise_strength"]),
                    )
                except (BudgetExceededError, InvalidActionError):
                    continue

                for child_id in child_ids:
                    if launched >= max_children:
                        break
                    if not self._can_afford(env, target_nfe, initial_budget, per_child_cost):
                        return
                    try:
                        self._forward_to_step(env, child_id, probe_step, solver="euler", cfg=None)
                        self._preview(env, child_id, target_nfe, initial_budget)
                    except (BudgetExceededError, InvalidActionError):
                        return
                    launched += 1
                    remaining -= 1
                    made_progress = True

                self._prune_losers(env, schedule)
            if not made_progress:
                return

    def _finish_best_active(
        self,
        env: FlowTTSEnv,
        finish_budget: int,
        target_nfe: int,
        initial_budget: int,
    ) -> None:
        remaining = max(0, int(finish_budget))
        while remaining > 0:
            best = self._best_live_preview(env)
            if best is None:
                return
            particle = self._particle(env, best.particle_id)
            if particle is None or particle.status != "active":
                return
            cost = self._move_cost_by_step(env, best.particle_id, len(env.time_grid) - 1) + 1
            if cost <= 1 or not self._can_afford(env, target_nfe, initial_budget, cost):
                return
            try:
                self._forward_to_step(env, best.particle_id, len(env.time_grid) - 1, solver="euler", cfg=None)
                self._preview(env, best.particle_id, target_nfe, initial_budget)
            except (BudgetExceededError, InvalidActionError):
                return
            remaining -= 1

    def _tail_actions(
        self,
        env: FlowTTSEnv,
        schedule: dict[str, float | int | bool | str],
        target_nfe: int,
        initial_budget: int,
    ) -> None:
        rounds = int(schedule["tail_rounds"])
        late_step = int(schedule["late_step"])
        while rounds > 0 and self._spent(env, initial_budget) < target_nfe and env.budget_left > 0:
            contenders = self._contender_previews(env, schedule)
            if not contenders:
                return

            chosen = None
            if len(contenders) >= 2 and self._preview_gap(contenders[0], contenders[1]) <= float(schedule["split_gap"]):
                runner_up = contenders[1]
                runner_particle = self._particle(env, runner_up.particle_id)
                if runner_particle is not None and runner_particle.status == "active":
                    chosen = runner_up
            if chosen is None:
                chosen = contenders[0]

            particle = self._particle(env, chosen.particle_id)
            if particle is not None and particle.status == "active":
                next_step = min(len(env.time_grid) - 1, self._time_to_step(env, particle.time) + 1)
                if next_step > self._time_to_step(env, particle.time):
                    cost = self._move_cost_by_step(env, chosen.particle_id, next_step) + 1
                    if self._can_afford(env, target_nfe, initial_budget, cost):
                        try:
                            self._forward_to_step(env, chosen.particle_id, next_step, solver="euler", cfg=None)
                            self._preview(env, chosen.particle_id, target_nfe, initial_budget)
                            rounds -= 1
                            continue
                        except (BudgetExceededError, InvalidActionError):
                            return

            best = self._best_live_preview(env)
            if best is None:
                return
            child_cost = self._child_probe_cost(env, int(schedule["branch_step"]), late_step)
            if self._anchor_needs_more_exploration(env, best, schedule) and self._can_afford(
                env,
                target_nfe,
                initial_budget,
                child_cost,
            ):
                try:
                    child_ids = env.backward(
                        best.id,
                        target_time=float(env.time_grid[int(schedule["branch_step"])]),
                        noise_policy=str(schedule["noise_policy"]),
                        num_children=1,
                        strength=float(schedule["noise_strength"]),
                    )
                    if not child_ids:
                        return
                    self._forward_to_step(env, child_ids[0], late_step, solver="euler", cfg=None)
                    self._preview(env, child_ids[0], target_nfe, initial_budget)
                    rounds -= 1
                    self._prune_losers(env, schedule)
                    continue
                except (BudgetExceededError, InvalidActionError):
                    return
            return

    def _contender_previews(
        self,
        env: FlowTTSEnv,
        schedule: dict[str, float | int | bool | str],
    ) -> list[PreviewRecord]:
        previews = self._ranked_previews(env)
        if not previews:
            return []
        best = float(previews[0].score or 0.0)
        keep: list[PreviewRecord] = []
        for index, preview in enumerate(previews):
            gap = best - float(preview.score or 0.0)
            if index < int(schedule["survivor_count"]):
                keep.append(preview)
                continue
            if gap <= float(schedule["hold_margin"]) or float(preview.uncertainty or 0.0) >= float(
                schedule["uncertainty_gate"]
            ):
                keep.append(preview)
        return keep

    def _branch_candidates(
        self,
        env: FlowTTSEnv,
        schedule: dict[str, float | int | bool | str],
    ) -> list[PreviewRecord]:
        contenders = self._contender_previews(env, schedule)
        if not contenders:
            return []

        best = contenders[0]
        candidates = [best]
        if len(contenders) >= 2 and self._preview_gap(best, contenders[1]) <= float(schedule["split_gap"]):
            candidates.append(contenders[1])
        return candidates

    def _should_probe_anchor(
        self,
        env: FlowTTSEnv,
        anchor: PreviewRecord,
        schedule: dict[str, float | int | bool | str],
        target_nfe: int,
        initial_budget: int,
    ) -> bool:
        branch_time = float(env.time_grid[int(schedule["branch_step"])])
        if float(anchor.time) + 1e-9 < branch_time:
            return False
        if self._anchor_needs_more_exploration(env, anchor, schedule):
            return True
        remaining = target_nfe - self._spent(env, initial_budget)
        return remaining >= self._child_probe_cost(
            env,
            int(schedule["branch_step"]),
            int(schedule["late_probe_step"]),
        ) + 2

    def _anchor_needs_more_exploration(
        self,
        env: FlowTTSEnv,
        anchor: PreviewRecord,
        schedule: dict[str, float | int | bool | str],
    ) -> bool:
        improvement = self._recent_improvement(env, anchor.particle_id)
        uncertainty = float(anchor.uncertainty or 0.0)
        gap = self._gap_to_next(env, anchor.id)
        return (
            improvement <= float(schedule["plateau_gate"])
            or uncertainty >= float(schedule["uncertainty_gate"])
            or gap <= float(schedule["split_gap"])
        )

    def _prune_losers(
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
            uncertain = float(preview.uncertainty or 0.0) >= float(schedule["uncertainty_gate"])
            close = gap <= float(schedule["hold_margin"])
            if index < int(schedule["survivor_count"]) or uncertain or close:
                keep_ids.append(preview.particle_id)

        prune_ids = [
            preview.particle_id
            for preview in previews
            if preview.particle_id not in keep_ids
            and self._is_active(env, preview.particle_id)
            and (best - float(preview.score or 0.0)) > float(schedule["prune_margin"])
        ]
        if prune_ids:
            try:
                env.prune(prune_ids)
            except InvalidActionError:
                return

    def _safe_answer(self, env: FlowTTSEnv) -> AnswerRecord:
        try:
            return env.answer(rule="best_preview_score")
        except (BudgetExceededError, InvalidActionError):
            return env.answer(rule="latest_active")

    def _forward_to_step(
        self,
        env: FlowTTSEnv,
        particle_id: int,
        target_step: int,
        solver: str,
        cfg: dict[str, float] | None,
    ) -> None:
        target_step = max(0, min(len(env.time_grid) - 1, int(target_step)))
        target_time = float(env.time_grid[target_step])
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

    def _best_live_preview(self, env: FlowTTSEnv) -> PreviewRecord | None:
        previews = self._ranked_previews(env)
        return previews[0] if previews else None

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

    def _recent_improvement(self, env: FlowTTSEnv, particle_id: int) -> float:
        previews = [
            preview
            for preview in env.get_state().previews.values()
            if preview.particle_id == particle_id and preview.score is not None
        ]
        previews.sort(key=lambda preview: preview.id)
        if len(previews) < 2:
            return 1.0
        return float(previews[-1].score or 0.0) - float(previews[-2].score or 0.0)

    def _gap_to_next(self, env: FlowTTSEnv, anchor_id: int) -> float:
        previews = self._ranked_previews(env)
        for index, preview in enumerate(previews):
            if preview.id != anchor_id:
                continue
            if index + 1 >= len(previews):
                return 1.0
            return self._preview_gap(preview, previews[index + 1])
        return 1.0

    def _preview_gap(self, left: PreviewRecord, right: PreviewRecord) -> float:
        return abs(float(left.score or 0.0) - float(right.score or 0.0))

    def _child_probe_cost(self, env: FlowTTSEnv, branch_step: int, probe_step: int) -> int:
        return max(1, int(probe_step) - int(branch_step)) + 1

    def _move_cost_by_step(self, env: FlowTTSEnv, particle_id: int, target_step: int) -> int:
        particle = self._particle(env, particle_id)
        if particle is None:
            return 0
        current_step = self._time_to_step(env, particle.time)
        return max(0, int(target_step) - current_step)

    def _time_to_step(self, env: FlowTTSEnv, target_time: float) -> int:
        grid = list(env.time_grid)
        for index, value in enumerate(grid):
            if float(value) + 1e-9 >= float(target_time):
                return index
        return len(grid) - 1

    def _particle(self, env: FlowTTSEnv, particle_id: int):
        return env.get_state().particles.get(particle_id)

    def _is_active(self, env: FlowTTSEnv, particle_id: int) -> bool:
        particle = self._particle(env, particle_id)
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
