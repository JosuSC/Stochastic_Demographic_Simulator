from typing import Optional


class Person:
    """
    Person model used by the demographic simulation.

    Age is computed on demand as:
        age = _birth_time_offset + (Person._current_sim_time - _sim_time_anchor)

    That gives the same result as storing an absolute birth time, but avoids
    extra floating-point drift over long runs.

    Grief tracking uses an absolute end time instead of a countdown, so there
    is nothing to update on every event.
    """
    _id_counter = 0
    _current_sim_time: float = 0.0  # Class-level simulation clock

    __slots__ = (
        'id', 'sex', '_birth_time_offset', '_sim_time_anchor',
        'is_alive', 'partner_id', 'grief_end_time',
        'desired_children', 'children_count', 'is_pregnant',
        'wants_partner', 'wants_children',
        'last_mortality_check_time', 'last_partner_desire_check_time',
        'last_child_desire_check_time',
    )

    def __init__(self, sex: str, age_years: float, is_alive: bool = True):
        self.id: int = Person._id_counter
        Person._id_counter += 1

        self.sex: str = sex
        # Keep enough info to rebuild the age whenever we need it.
        self._birth_time_offset: float = age_years
        self._sim_time_anchor: float = Person._current_sim_time
        self.is_alive: bool = is_alive
        self.partner_id: Optional[int] = None

        # Absolute end of the grief period; 0.0 means the person is not grieving.
        self.grief_end_time: float = 0.0

        self.desired_children: int = 0
        self.children_count: int = 0
        self.is_pregnant: bool = False
        self.wants_partner: bool = False
        self.wants_children: bool = False

        self.last_mortality_check_time: float = Person._current_sim_time
        self.last_partner_desire_check_time: float = Person._current_sim_time
        self.last_child_desire_check_time: float = Person._current_sim_time

    # --- Age handling ---

    @property
    def age_years(self) -> float:
        """Return the current age based on the simulation clock."""
        return self._birth_time_offset + (Person._current_sim_time - self._sim_time_anchor)

    @age_years.setter
    def age_years(self, value: float) -> None:
        """Reset the age while keeping the current simulation time as reference."""
        self._birth_time_offset = value
        self._sim_time_anchor = Person._current_sim_time

    @property
    def age_months(self) -> float:
        return self.age_years * 12.0

    def is_grieving(self, current_time: float) -> bool:
        """Check whether the person is still inside the grief window."""
        return current_time < self.grief_end_time

    @classmethod
    def set_sim_time(cls, time: float) -> None:
        """Update the shared simulation clock used by all people."""
        cls._current_sim_time = time

    def __str__(self) -> str:
        return f"Person #{self.id} ({self.sex}, {self.age_years:.1f} yrs)"
