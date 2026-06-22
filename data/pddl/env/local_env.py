import re
from dataclasses import dataclass
from typing import Any


@dataclass
class LocalStep:
    observation: str
    reward: float
    done: bool
    valid: bool


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().rstrip(".").lower())


def _ids_from_subgoals(subgoals: list[str], prefix: str) -> list[int]:
    ids = {
        int(match.group(1))
        for subgoal in subgoals
        for match in re.finditer(rf"\b{re.escape(prefix)}(\d+)\b", subgoal.lower())
    }
    return sorted(ids)


class BaseLocalPDDLEnv:
    """Small deterministic backend for LatentMem PDDL domains absent from stock pddlgym."""

    def __init__(self, metadata: dict[str, Any]):
        self.metadata = metadata
        self.subgoals = list(metadata.get("subgoals") or [])
        self.goal_text = metadata.get("goal") or (
            "The goal is to satisfy the following conditions: " + " ".join(self.subgoals)
        )
        self.reward = 0.0
        self.done = False

    def reset(self) -> None:
        self.reward = self._score()
        self.done = self.reward >= 1.0

    def state_text(self) -> str:
        raise NotImplementedError

    def full_state_text(self) -> str:
        return self.state_text()

    def action_space(self) -> list[str]:
        raise NotImplementedError

    def step(self, action: str) -> LocalStep:
        raise NotImplementedError

    def _score(self) -> float:
        if not self.subgoals:
            return 0.0
        satisfied = sum(1 for subgoal in self.subgoals if self._subgoal_satisfied(subgoal))
        return satisfied / len(self.subgoals)

    def _finish(self, observation: str, valid: bool = True) -> LocalStep:
        self.reward = max(self.reward, self._score())
        self.done = self.reward >= 1.0
        if self.done and "goal is satisfied" not in observation.lower():
            observation = f"{observation} The goal is satisfied."
        return LocalStep(observation, self.reward, self.done, valid)

    def _subgoal_satisfied(self, subgoal: str) -> bool:
        raise NotImplementedError


class LocalBarmanEnv(BaseLocalPDDLEnv):
    def __init__(self, metadata: dict[str, Any]):
        super().__init__(metadata)
        shot_ids = _ids_from_subgoals(self.subgoals, "shot")
        self.shot_ids = list(range(1, max(shot_ids or [4]) + 1))
        self.ingredients = ["ingredient1", "ingredient2", "ingredient3"]
        self.cocktails = [f"cocktail{i}" for i in range(1, 7)]
        self.hands: dict[str, str | None] = {"left": None, "right": None}
        self.shots: dict[str, dict[str, Any]] = {}
        self.shaker = {
            "on_table": True,
            "held_by": None,
            "ingredients": [],
            "cocktail": None,
            "clean": True,
        }

    def reset(self) -> None:
        self.hands = {"left": None, "right": None}
        self.shots = {
            f"shot{i}": {"contains": None, "clean": True, "on_table": True, "held_by": None}
            for i in self.shot_ids
        }
        self.shaker = {
            "on_table": True,
            "held_by": None,
            "ingredients": [],
            "cocktail": None,
            "clean": True,
        }
        super().reset()

    def state_text(self) -> str:
        parts = ["Left hand is empty." if self.hands["left"] is None else f"Left hand holds {self.hands['left']}."]
        parts.append("Right hand is empty." if self.hands["right"] is None else f"Right hand holds {self.hands['right']}.")
        for ingredient in self.ingredients:
            parts.append(f"Dispenser{ingredient[-1]} dispenses {ingredient}.")
        for cocktail in self.cocktails:
            parts.append(f"{cocktail.capitalize()} can be made by shaking two ingredients in the shaker.")
        for shot, state in sorted(self.shots.items()):
            if state["contains"]:
                parts.append(f"{shot.capitalize()} contains {state['contains']}.")
            else:
                parts.append(f"{shot.capitalize()} is empty.")
            if state["clean"]:
                parts.append(f"{shot.capitalize()} is clean.")
            if state["on_table"]:
                parts.append(f"{shot.capitalize()} is on the table.")
        if self.shaker["cocktail"]:
            parts.append(f"Shaker1 contains {self.shaker['cocktail']}.")
            parts.append("Shaker1 is shaked.")
        elif self.shaker["ingredients"]:
            for ingredient in self.shaker["ingredients"]:
                parts.append(f"Shaker1 contains {ingredient}.")
            parts.append("Shaker1 is unshaked.")
        else:
            parts.append("Shaker1 is empty.")
        if self.shaker["clean"]:
            parts.append("Shaker1 is clean.")
        if self.shaker["on_table"]:
            parts.append("Shaker1 is on the table.")
        return " ".join(parts)

    def action_space(self) -> list[str]:
        actions = ["check valid actions", "look around"]
        for hand, held in self.hands.items():
            if held is None:
                for shot, state in sorted(self.shots.items()):
                    if state["on_table"]:
                        actions.append(f"{hand} grasp {shot}.")
                if self.shaker["on_table"]:
                    actions.append(f"{hand} grasp shaker1.")
            else:
                actions.append(f"{hand} leave {held}.")
        for shot, state in sorted(self.shots.items()):
            if state["held_by"] and state["contains"] is None:
                for ingredient in self.ingredients:
                    actions.append(
                        f"fill-shot glass {shot} with {ingredient} with {state['held_by']} and "
                        f"{self._other_hand(state['held_by'])} holding dispenser{ingredient[-1]}."
                    )
            if state["held_by"] and state["contains"]:
                actions.append(
                    f"pour-shot-to-used-shaker from a shot glass {shot} with {state['contains']} "
                    f"to a used shaker shaker1 with hand {state['held_by']} from level l0 to level l1."
                )
                actions.append(f"clean-shot glass {shot} with {state['contains']} with hand {state['held_by']} holding shot glass and {self._other_hand(state['held_by'])}.")
                actions.append(f"use hand {state['held_by']} to empty-shot glass {shot} with beverage {state['contains']}.")
        if self.shaker["ingredients"]:
            for cocktail in self.cocktails:
                actions.append(f"shake a cocktail {cocktail} with ingredient ingredient1 and ingredient ingredient2 in a shaker shaker1 with hand right and hand left.")
        if self.shaker["cocktail"]:
            for shot in sorted(self.shots):
                actions.append(f"pour-shaker-to-shot to a shot glass {shot} the ingredient {self.shaker['cocktail']} with hand right from shaker shaker1 from level l2 to level l1.")
        return actions

    def step(self, action: str) -> LocalStep:
        text = action.lower()
        hand = self._hand_from_text(text)
        shot = self._shot_from_text(text)
        ingredient = self._ingredient_from_text(text)
        cocktail = self._cocktail_from_text(text)

        if "grasp" in text:
            target = shot or ("shaker1" if "shaker" in text else None)
            hand = hand or self._free_hand()
            if hand is None or target is None or self.hands[hand] is not None:
                return self._invalid("Cannot grasp that container now.")
            if target == "shaker1":
                if not self.shaker["on_table"]:
                    return self._invalid("Shaker1 is not on the table.")
                self.shaker["on_table"] = False
                self.shaker["held_by"] = hand
            else:
                if target not in self.shots or not self.shots[target]["on_table"]:
                    return self._invalid("The shot glass is not on the table.")
                self.shots[target]["on_table"] = False
                self.shots[target]["held_by"] = hand
            self.hands[hand] = target
            return self._finish(f"You are holding {target} with {hand}.")

        if "leave" in text:
            target = shot or ("shaker1" if "shaker" in text else None)
            hand = hand or self._holding_hand(target)
            if hand is None or target is None or self.hands.get(hand) != target:
                return self._invalid("You are not holding that container.")
            self.hands[hand] = None
            if target == "shaker1":
                self.shaker["on_table"] = True
                self.shaker["held_by"] = None
            else:
                self.shots[target]["on_table"] = True
                self.shots[target]["held_by"] = None
            return self._finish(f"{target.capitalize()} is on the table. {hand.capitalize()} hand is empty.")

        if "fill-shot" in text or ("fill" in text and shot and ingredient):
            if shot not in self.shots or ingredient is None or self.shots[shot]["contains"] is not None:
                return self._invalid("The shot glass cannot be filled now.")
            self.shots[shot]["contains"] = ingredient
            self.shots[shot]["clean"] = False
            return self._finish(f"{shot.capitalize()} contains {ingredient}.")

        if "empty-shot" in text and shot:
            if shot not in self.shots or self.shots[shot]["contains"] is None:
                return self._invalid("The shot glass is already empty.")
            old = self.shots[shot]["contains"]
            self.shots[shot]["contains"] = None
            return self._finish(f"{shot.capitalize()} no longer contains {old}.")

        if "clean-shot" in text and shot:
            if shot not in self.shots:
                return self._invalid("Unknown shot glass.")
            self.shots[shot]["contains"] = None
            self.shots[shot]["clean"] = True
            return self._finish(f"{shot.capitalize()} is clean.")

        if "pour-shot-to" in text and shot:
            if shot not in self.shots or self.shots[shot]["contains"] is None:
                return self._invalid("The shot glass does not contain anything to pour.")
            poured = self.shots[shot]["contains"]
            self.shots[shot]["contains"] = None
            self.shaker["ingredients"].append(poured)
            self.shaker["clean"] = False
            self.shaker["cocktail"] = None
            return self._finish(f"Shaker1 contains {poured}. {shot.capitalize()} is empty.")

        if "shake" in text and cocktail:
            if not self.shaker["ingredients"]:
                return self._invalid("The shaker has no ingredients.")
            self.shaker["cocktail"] = cocktail
            return self._finish(f"Shaker1 contains {cocktail}. Shaker1 is shaked.")

        if "pour-shaker-to-shot" in text and shot:
            if shot not in self.shots or self.shaker["cocktail"] is None:
                return self._invalid("The shaker does not contain a cocktail.")
            cocktail = cocktail or self.shaker["cocktail"]
            self.shots[shot]["contains"] = cocktail
            self.shots[shot]["clean"] = False
            self.shaker["cocktail"] = None
            self.shaker["ingredients"] = []
            return self._finish(f"{shot.capitalize()} contains {cocktail}.")

        return self._invalid("The action is not valid and therefore takes no effect.")

    def _invalid(self, observation: str) -> LocalStep:
        return LocalStep(observation, self.reward, self.done, False)

    def _subgoal_satisfied(self, subgoal: str) -> bool:
        match = re.search(r"\b(shot\d+)\s+contains\s+(\w+)", subgoal.lower())
        if not match:
            return False
        shot, content = match.groups()
        return self.shots.get(shot, {}).get("contains") == content

    @staticmethod
    def _other_hand(hand: str) -> str:
        return "left" if hand == "right" else "right"

    def _free_hand(self) -> str | None:
        for hand, held in self.hands.items():
            if held is None:
                return hand
        return None

    def _holding_hand(self, target: str | None) -> str | None:
        if target is None:
            return None
        for hand, held in self.hands.items():
            if held == target:
                return hand
        return None

    @staticmethod
    def _hand_from_text(text: str) -> str | None:
        if "left" in text:
            return "left"
        if "right" in text:
            return "right"
        return None

    @staticmethod
    def _shot_from_text(text: str) -> str | None:
        match = re.search(r"\bshot\d+\b", text)
        return match.group(0) if match else None

    @staticmethod
    def _ingredient_from_text(text: str) -> str | None:
        match = re.search(r"\bingredient\d+\b", text)
        return match.group(0) if match else None

    @staticmethod
    def _cocktail_from_text(text: str) -> str | None:
        match = re.search(r"\bcocktail\d+\b", text)
        return match.group(0) if match else None


class LocalTyreworldEnv(BaseLocalPDDLEnv):
    def __init__(self, metadata: dict[str, Any]):
        super().__init__(metadata)
        wheel_ids = sorted(set(_ids_from_subgoals(self.subgoals, "r") + _ids_from_subgoals(self.subgoals, "w")))
        self.wheel_ids = list(range(1, max(wheel_ids or [1]) + 1))
        self.boot_open = False
        self.holding: set[str] = set()
        self.in_boot: set[str] = set()
        self.inflated: set[str] = set()
        self.hubs: dict[str, dict[str, Any]] = {}

    def reset(self) -> None:
        self.boot_open = False
        self.holding = set()
        self.in_boot = {"wrench", "jack", "pump"} | {f"r{i}" for i in self.wheel_ids}
        self.inflated = set()
        self.hubs = {
            f"the-hub{i}": {
                "wheel": f"w{i}",
                "tight": True,
                "fastened": True,
                "on_ground": True,
            }
            for i in self.wheel_ids
        }
        super().reset()

    def state_text(self) -> str:
        parts = [f"Boot is {'open' if self.boot_open else 'closed'}.", "Boot is unlocked."]
        for obj in sorted(self.in_boot):
            parts.append(f"{obj.capitalize()} is in boot.")
        for obj in sorted(self.holding):
            parts.append(f"You have {obj}.")
        for hub, state in sorted(self.hubs.items()):
            parts.append(f"Hub {hub} is {'on' if state['on_ground'] else 'not on'} the ground.")
            parts.append(f"Hub {hub} is {'fastened' if state['fastened'] else 'unfastened'}.")
            parts.append(f"The nut nuts{hub.replace('the-hub', '')} on the hub {hub} is {'tight' if state['tight'] else 'loose'}.")
            if state["wheel"]:
                parts.append(f"{state['wheel']} is on {hub}.")
        for i in self.wheel_ids:
            parts.append(f"Wheel r{i} is intact.")
            if f"r{i}" in self.inflated:
                parts.append(f"Wheel r{i} is inflated.")
            else:
                parts.append(f"Wheel r{i} is not inflated.")
        return " ".join(parts)

    def action_space(self) -> list[str]:
        actions = ["check valid actions", "look around"]
        if not self.boot_open:
            actions.append("Open boot.")
        else:
            actions.append("Close boot.")
            for obj in sorted(self.in_boot):
                actions.append(f"Fetch {obj} from boot.")
            for obj in sorted(self.holding):
                actions.append(f"Put-away {obj} in boot.")
        for i in self.wheel_ids:
            hub = f"the-hub{i}"
            state = self.hubs[hub]
            if "wrench" in self.holding and state["tight"] and state["on_ground"]:
                actions.append(f"Loosen the nut nuts{i} on the hub {hub}.")
            if "jack" in self.holding and state["on_ground"]:
                actions.append(f"jack-up the hub {hub}.")
            if "wrench" in self.holding and not state["on_ground"] and state["fastened"] and not state["tight"]:
                actions.append(f"Undo the fastening of the nut nuts{i} on the hub {hub}.")
            if not state["on_ground"] and not state["fastened"] and state["wheel"] == f"w{i}":
                actions.append(f"Remove-wheel w{i} from the hub {hub}.")
            if not state["on_ground"] and not state["fastened"] and state["wheel"] is None and f"r{i}" in self.holding:
                actions.append(f"put-on-wheel r{i} on the hub {hub}.")
            if "pump" in self.holding and f"r{i}" not in self.inflated:
                actions.append(f"Inflate the wheel r{i}.")
        return actions

    def step(self, action: str) -> LocalStep:
        text = action.lower()
        obj = self._object_from_text(text)
        hub = self._hub_from_text(text)
        idx = self._idx_from_hub_or_object(hub, obj, text)

        if "open" in text and "boot" in text:
            self.boot_open = True
            return self._finish("Boot is open.")
        if "close" in text and "boot" in text:
            self.boot_open = False
            return self._finish("Boot is closed.")
        if "fetch" in text and obj:
            if not self.boot_open or obj not in self.in_boot:
                return self._invalid("The object cannot be fetched from boot now.")
            self.in_boot.remove(obj)
            self.holding.add(obj)
            return self._finish(f"You have {obj}.")
        if "put-away" in text and obj:
            if not self.boot_open or obj not in self.holding:
                return self._invalid("The object cannot be put away now.")
            self.holding.remove(obj)
            self.in_boot.add(obj)
            return self._finish(f"{obj.capitalize()} is in boot.")
        if "loosen" in text and idx:
            state = self.hubs[f"the-hub{idx}"]
            if "wrench" not in self.holding or not state["tight"]:
                return self._invalid("The nut cannot be loosened now.")
            state["tight"] = False
            return self._finish(f"The nut nuts{idx} on the hub the-hub{idx} is loose.")
        if "jack-up" in text and idx:
            state = self.hubs[f"the-hub{idx}"]
            if "jack" not in self.holding or not state["on_ground"]:
                return self._invalid("The hub cannot be jacked up now.")
            state["on_ground"] = False
            return self._finish(f"Hub the-hub{idx} is not on the ground.")
        if "undo" in text and idx:
            state = self.hubs[f"the-hub{idx}"]
            if "wrench" not in self.holding or state["on_ground"] or not state["fastened"]:
                return self._invalid("The fastening cannot be undone now.")
            state["fastened"] = False
            return self._finish(f"Hub the-hub{idx} is unfastened.")
        if "remove-wheel" in text and idx:
            state = self.hubs[f"the-hub{idx}"]
            wheel = obj or state["wheel"]
            if state["on_ground"] or state["fastened"] or state["wheel"] != wheel:
                return self._invalid("The wheel cannot be removed now.")
            state["wheel"] = None
            self.holding.add(wheel)
            return self._finish(f"You have {wheel}.")
        if "put-on-wheel" in text and idx:
            state = self.hubs[f"the-hub{idx}"]
            wheel = obj or f"r{idx}"
            if state["on_ground"] or state["fastened"] or state["wheel"] is not None or wheel not in self.holding:
                return self._invalid("The wheel cannot be put on now.")
            state["wheel"] = wheel
            self.holding.remove(wheel)
            return self._finish(f"{wheel} is on the-hub{idx}.")
        if "inflate" in text and obj:
            if "pump" not in self.holding or not obj.startswith("r"):
                return self._invalid("The wheel cannot be inflated now.")
            self.inflated.add(obj)
            return self._finish(f"Wheel {obj} is inflated.")
        if "do-up" in text and idx:
            state = self.hubs[f"the-hub{idx}"]
            if "wrench" not in self.holding or state["fastened"]:
                return self._invalid("The hub cannot be fastened now.")
            state["fastened"] = True
            return self._finish(f"Hub the-hub{idx} is fastened.")
        if "tighten" in text and idx:
            state = self.hubs[f"the-hub{idx}"]
            if "wrench" not in self.holding:
                return self._invalid("The nut cannot be tightened now.")
            state["tight"] = True
            return self._finish(f"The nut nuts{idx} on the hub the-hub{idx} is tight.")
        if "jack-down" in text and idx:
            state = self.hubs[f"the-hub{idx}"]
            state["on_ground"] = True
            return self._finish(f"Hub the-hub{idx} is on the ground.")

        return self._invalid("The action is not valid and therefore takes no effect.")

    def _invalid(self, observation: str) -> LocalStep:
        return LocalStep(observation, self.reward, self.done, False)

    def _subgoal_satisfied(self, subgoal: str) -> bool:
        text = _norm(subgoal)
        match = re.search(r"wheel (r\d+) is inflated", text)
        if match:
            return match.group(1) in self.inflated
        match = re.search(r"\b(r\d+) is on (the-hub\d+)", text)
        if match:
            wheel, hub = match.groups()
            return self.hubs.get(hub, {}).get("wheel") == wheel
        match = re.search(r"\b(w\d+) is in boot", text)
        if match:
            return match.group(1) in self.in_boot
        return False

    @staticmethod
    def _object_from_text(text: str) -> str | None:
        for pattern in [r"\b(?:r|w)\d+\b", r"\bwrench\b", r"\bjack\b", r"\bpump\b"]:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        return None

    @staticmethod
    def _hub_from_text(text: str) -> str | None:
        match = re.search(r"\bthe-hub\d+\b", text)
        return match.group(0) if match else None

    @staticmethod
    def _idx_from_hub_or_object(hub: str | None, obj: str | None, text: str) -> int | None:
        for candidate in [hub, obj]:
            if candidate:
                match = re.search(r"(\d+)", candidate)
                if match:
                    return int(match.group(1))
        match = re.search(r"\bnuts(\d+)\b", text)
        return int(match.group(1)) if match else None


def make_local_pddl_env(game_name: str, metadata: dict[str, Any]) -> BaseLocalPDDLEnv:
    if game_name == "barman":
        return LocalBarmanEnv(metadata)
    if game_name == "tyreworld":
        return LocalTyreworldEnv(metadata)
    raise ValueError(f"No local PDDL backend for {game_name!r}")

