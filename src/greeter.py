"""
Greeter module: confidence-weighted emotion aggregation + per-identity
session state machine. Emits one curated greeting per check-in.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import random
import time

import numpy as np
import yaml


class State(Enum):
    IDLE       = "idle"        # not currently engaging with this identity
    COLLECTING = "collecting"  # accumulating probability vectors
    EMITTED    = "emitted"     # greeting fired, banner is on screen
    COOLDOWN   = "cooldown"    # banner gone; don't re-fire for this identity yet


@dataclass
class Greeting:
    name: str
    emotion: str
    dominance: float
    message: str
    emitted_at: float


@dataclass
class IdentitySession:
    name: str
    state: State = State.IDLE
    prob_buffer: list = field(default_factory=list)
    state_entered_at: float = 0.0
    last_seen_at: float = 0.0
    last_greeting: Optional[Greeting] = None


class GreetingAggregator:
    def __init__(
        self,
        window_frames: int,
        dominance_threshold: float,
        cooldown_seconds: float,
        display_duration: float,
        stale_after: float,
        emotion_labels: list,
        messages: dict,
    ):
        self.window_frames = window_frames
        self.dominance_threshold = dominance_threshold
        self.cooldown_seconds = cooldown_seconds
        self.display_duration = display_duration
        self.stale_after = stale_after
        self.emotion_labels = emotion_labels
        self.messages = messages
        self._sessions: dict = {}
        self._active_greeting: Optional[Greeting] = None

    def observe(self, name: str, probs: np.ndarray) -> None:
        now = time.monotonic()
        session = self._sessions.get(name)
        if session is None:
            session = IdentitySession(name=name, state_entered_at=now)
            self._sessions[name] = session

        self._tick(session, now)
        session.last_seen_at = now

        if session.state == State.IDLE:
            session.state = State.COLLECTING
            session.state_entered_at = now
            session.prob_buffer = [probs.copy()]
        elif session.state == State.COLLECTING:
            session.prob_buffer.append(probs.copy())
            if len(session.prob_buffer) >= self.window_frames:
                self._active_greeting = self._emit(session, now)

    def current_greeting(self) -> Optional[Greeting]:
        if self._active_greeting is None:
            return None
        now = time.monotonic()
        if now - self._active_greeting.emitted_at >= self.display_duration:
            session = self._sessions.get(self._active_greeting.name)
            if session is not None and session.state == State.EMITTED:
                session.state = State.COOLDOWN
                session.state_entered_at = self._active_greeting.emitted_at + self.display_duration
            self._active_greeting = None
            return None
        return self._active_greeting

    def reset(self, name: Optional[str] = None) -> None:
        if name is None:
            self._sessions.clear()
            self._active_greeting = None
            return
        self._sessions.pop(name, None)
        if self._active_greeting is not None and self._active_greeting.name == name:
            self._active_greeting = None

    def _tick(self, session: IdentitySession, now: float) -> None:
        if session.state == State.EMITTED:
            if now - session.state_entered_at >= self.display_duration:
                session.state = State.COOLDOWN
                session.state_entered_at = now
        if session.state == State.COOLDOWN:
            if now - session.state_entered_at >= self.cooldown_seconds:
                session.state = State.IDLE
                session.state_entered_at = now
                session.prob_buffer = []
        if session.state == State.COLLECTING:
            if now - session.last_seen_at >= self.stale_after:
                session.state = State.IDLE
                session.state_entered_at = now
                session.prob_buffer = []

    def _emit(self, session: IdentitySession, now: float) -> Greeting:
        emotion, dominance = _aggregate(
            session.prob_buffer,
            self.emotion_labels,
            self.dominance_threshold,
        )
        message = _pick_message(emotion, session.name, self.messages)
        greeting = Greeting(
            name=session.name,
            emotion=emotion,
            dominance=dominance,
            message=message,
            emitted_at=now,
        )
        session.state = State.EMITTED
        session.state_entered_at = now
        session.last_greeting = greeting
        session.prob_buffer = []
        return greeting


def _aggregate(prob_buffer, labels, dominance_threshold):
    """
    Confidence-weighted aggregation: sum the probability vectors,
    normalise to a distribution, pick argmax. Fall back to 'unclear'
    when the dominant class share is below threshold.
    """
    stacked = np.stack(prob_buffer, axis=0)
    summed = stacked.sum(axis=0)
    total = float(summed.sum())
    if total <= 0:
        return "unclear", 0.0
    normalized = summed / total
    top_idx = int(np.argmax(normalized))
    dominance = float(normalized[top_idx])
    if dominance < dominance_threshold:
        return "unclear", dominance
    return labels[top_idx], dominance


def _pick_message(emotion, name, messages):
    templates = messages.get(emotion) or messages.get("unclear") or ["Welcome, {name}."]
    return random.choice(templates).format(name=name)


def _build_from_config() -> GreetingAggregator:
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    g = config["greeting"]
    with open(g["messages_path"]) as f:
        messages = yaml.safe_load(f)["greetings"]
    return GreetingAggregator(
        window_frames=g["window_frames"],
        dominance_threshold=g["dominance_threshold"],
        cooldown_seconds=g["cooldown_seconds"],
        display_duration=g["display_duration"],
        stale_after=g["stale_after"],
        emotion_labels=config["emotions"]["labels"],
        messages=messages,
    )


_aggregator = _build_from_config()


def observe(name: str, probs: np.ndarray) -> None:
    _aggregator.observe(name, probs)


def current_greeting() -> Optional[Greeting]:
    return _aggregator.current_greeting()


def reset(name: Optional[str] = None) -> None:
    _aggregator.reset(name)
