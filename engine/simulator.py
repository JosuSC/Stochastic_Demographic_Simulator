from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from models.person import Person
from utils.probability import get_uniform, evaluate_probability, get_exponential


class EventType(str, Enum):
    """Tipos de eventos base del motor DES."""
    DEATH_EVENT = "DEATH_EVENT"
    PARTNER_DESIRE_EVENT = "PARTNER_DESIRE_EVENT"
    COUPLE_FORMATION_EVENT = "COUPLE_FORMATION_EVENT"
    CHILD_DESIRE_EVENT = "CHILD_DESIRE_EVENT"
    PREGNANCY_ATTEMPT_EVENT = "PREGNANCY_ATTEMPT_EVENT"
    PREGNANCY_START_EVENT = "PREGNANCY_START_EVENT"
    BIRTH_EVENT = "BIRTH_EVENT"
    BREAKUP_EVENT = "BREAKUP_EVENT"

    EPIDEMIC_EVENT = "EPIDEMIC_EVENT"
    WAR_EVENT = "WAR_EVENT"
    DISASTER_EVENT = "DISASTER_EVENT"
    CURE_EVENT = "CURE_EVENT"
    CRISIS_EVENT = "CRISIS_EVENT"
    BABY_BOOM_EVENT = "BABY_BOOM_EVENT"
    ACCIDENT_EVENT = "ACCIDENT_EVENT"


@dataclass(order=True)
class Event:
    """Evento ordenable para la lista futura (FEL)."""
    time: float
    seq: int
    event_type: EventType = field(compare=False)
    target_id: Optional[int] = field(compare=False, default=None)
    payload: Dict[str, int | float | str] = field(compare=False, default_factory=dict)


@dataclass
class GestationRecord:
    """Registro minimo de gestacion."""
    fetus_id: int
    mother_id: int
    father_id: Optional[int]
    due_time: float


@dataclass
class RelationshipRecord:
    """Estado temporal de una pareja."""
    partner_a: int
    partner_b: int
    last_breakup_check_time: float
    last_pregnancy_attempt_time: float


class Simulator:
    """Motor DES con FEL y tiempo continuo en años."""

    DEATH_CHECK_RATE_PER_YEAR = 12.0
    PARTNER_DESIRE_RATE_PER_YEAR = 12.0
    CHILD_DESIRE_RATE_PER_YEAR = 1.0
    COUPLE_FORMATION_RATE_PER_YEAR = 6.0
    PREGNANCY_ATTEMPT_RATE_PER_YEAR = 12.0
    BREAKUP_CHECK_RATE_PER_YEAR = 12.0

    EPIDEMIC_RATE_PER_YEAR = 0.01
    WAR_RATE_PER_YEAR = 0.005
    DISASTER_RATE_PER_YEAR = 0.01
    CRISIS_RATE_PER_YEAR = 0.03
    BABY_BOOM_RATE_PER_YEAR = 0.03
    CURE_RATE_PER_YEAR = 0.005
    ACCIDENT_RATE_PER_YEAR = 0.05

    def __init__(self, initial_females: int, initial_males: int):
        """Arma el estado inicial del simulador y crea la poblacion base."""
        self.current_time_years: float = 0.0
        self.current_month: float = 0.0
        self.total_births: int = 0
        self.total_deaths: int = 0

        self.people: Dict[int, Person] = {}
        self.alive_ids: Set[int] = set()
        self.dead_ids: Set[int] = set()
        self.male_group: Set[int] = set()
        self.female_group: Set[int] = set()
        self.gestation_group: Dict[int, GestationRecord] = {}
        self.mother_gestations: Dict[int, Set[int]] = {}
        self.relationships: Dict[Tuple[int, int], RelationshipRecord] = {}
        self.ready_males: Set[int] = set()
        self.ready_females: Set[int] = set()

        self.fel: List[Event] = []
        self._event_seq: int = 0
        self._schedule_built: bool = False

        self.current_year_logs: List[str] = []

        self.disease_cure_active: bool = False
        self.economic_crisis_active_years: float = 0.0
        self.baby_boom_active_years: float = 0.0

        self.config = {
            "crisis_economica_enabled": False,
            "baby_boom_enabled": False,
            "cura_enfermedades_enabled": False,
            "accidentes_masivo": False,
        }

        self._initialize_population(initial_females, "F")
        self._initialize_population(initial_males, "M")

    def _log_event(self, message: str) -> None:
        """Añade un evento al registro del año actual."""
        self.current_year_logs.append(message)

    def _annual_to_monthly_prob(self, annual_prob: float) -> float:
        """Convierte una probabilidad anual a mensual usando complementos."""
        if annual_prob >= 1.0:
            return 1.0
        if annual_prob <= 0.0:
            return 0.0
        return 1.0 - math.pow(1.0 - annual_prob, 1.0 / 12.0)

    def _period_prob(self, monthly_prob: float, delta_years: float) -> float:
        """Ajusta una probabilidad mensual al periodo real transcurrido."""
        if monthly_prob <= 0.0:
            return 0.0
        if monthly_prob >= 1.0:
            return 1.0
        if delta_years <= 0.0:
            return 0.0
        delta_months = delta_years * 12.0
        return 1.0 - math.pow(1.0 - monthly_prob, delta_months)

    def _initialize_population(self, count: int, sex: str) -> None:
        """Crea la poblacion inicial con edades uniformes entre 0 y 100 años."""
        for _ in range(count):
            age_years = get_uniform(0, 100)
            person = Person(sex, age_years)
            person.desired_children = self._get_desired_children()
            self._add_person_to_population(person)

    def _add_person_to_population(self, person: Person) -> None:
        """Registra una persona viva en los indices del simulador."""
        self.people[person.id] = person
        self.alive_ids.add(person.id)
        if person.sex == "M":
            self.male_group.add(person.id)
        else:
            self.female_group.add(person.id)

    def get_alive_population(self) -> List[Person]:
        """Devuelve la lista de vivos para reportes externos."""
        return [self.people[pid] for pid in self.alive_ids]

    def _get_desired_children(self) -> int:
        """Sortea cuantos hijos desea tener una persona segun la tabla dada."""
        r = random.random()
        if r < 0.292:
            return 1
        if r < 0.658:
            return 2
        if r < 0.829:
            return 3
        if r < 0.926:
            return 4
        if r < 0.975:
            return 5
        return 6

    def _get_grief_time_years(self, age_years: float) -> float:
        """Devuelve el tiempo de duelo esperado en años."""
        if age_years <= 15:
            mean_months = 3
        elif age_years <= 21:
            mean_months = 6
        elif age_years <= 35:
            mean_months = 6
        elif age_years <= 45:
            mean_months = 12
        elif age_years <= 60:
            mean_months = 24
        else:
            mean_months = 48
        return max(1.0 / 12.0, get_exponential(mean_months) / 12.0)

    def _wants_partner(self, person: Person) -> bool:
        """Decide si alguien busca pareja en este momento."""
        if person.partner_id is not None:
            return False
        if person.grief_time_remaining_years > 0:
            return False

        age = person.age_years
        if age < 12:
            prob = 0.0
        elif age <= 15:
            prob = 0.60
        elif age <= 21:
            prob = 0.65
        elif age <= 35:
            prob = 0.80
        elif age <= 45:
            prob = 0.60
        elif age <= 60:
            prob = 0.50
        else:
            prob = 0.20
        return evaluate_probability(prob)

    def _match_probability(self, p1: Person, p2: Person) -> float:
        """Devuelve probabilidad de match segun edad y diferencia."""
        is_p1_minor = p1.age_years < 18
        is_p2_minor = p2.age_years < 18
        if (is_p1_minor and not is_p2_minor) or (is_p2_minor and not is_p1_minor):
            return 0.0

        diff = abs(p1.age_years - p2.age_years)
        if diff <= 5:
            return 0.45
        if diff <= 10:
            return 0.40
        if diff <= 15:
            return 0.35
        if diff <= 20:
            return 0.25
        return 0.15

    def _pair_key(self, a_id: int, b_id: int) -> Tuple[int, int]:
        """Normaliza el par de ids para diccionarios."""
        return (min(a_id, b_id), max(a_id, b_id))

    def schedule_event(
        self,
        time: float,
        event_type: EventType,
        target_id: Optional[int] = None,
        payload: Optional[Dict[str, int | float | str]] = None,
    ) -> None:
        """Agenda un evento en la FEL."""
        if payload is None:
            payload = {}
        if time < self.current_time_years:
            time = self.current_time_years
        event = Event(time=time, seq=self._event_seq, event_type=event_type, target_id=target_id, payload=payload)
        self._event_seq += 1
        heapq.heappush(self.fel, event)

    def build_initial_schedule(self) -> None:
        """Construye las sub-agendas iniciales y las inserta en la FEL."""
        if self._schedule_built:
            return
        self._schedule_built = True

        for person_id in list(self.alive_ids):
            person = self.people[person_id]
            self.schedule_event(
                self.current_time_years + random.expovariate(self.DEATH_CHECK_RATE_PER_YEAR),
                EventType.DEATH_EVENT,
                target_id=person.id,
            )
            self.schedule_event(
                self.current_time_years + random.expovariate(self.PARTNER_DESIRE_RATE_PER_YEAR),
                EventType.PARTNER_DESIRE_EVENT,
                target_id=person.id,
            )
            self.schedule_event(
                self.current_time_years + random.expovariate(self.CHILD_DESIRE_RATE_PER_YEAR),
                EventType.CHILD_DESIRE_EVENT,
                target_id=person.id,
            )

        self.schedule_event(
            self.current_time_years + random.expovariate(self.COUPLE_FORMATION_RATE_PER_YEAR),
            EventType.COUPLE_FORMATION_EVENT,
        )

        self._schedule_next_global_event(EventType.EPIDEMIC_EVENT, self.EPIDEMIC_RATE_PER_YEAR)
        self._schedule_next_global_event(EventType.WAR_EVENT, self.WAR_RATE_PER_YEAR)
        self._schedule_next_global_event(EventType.DISASTER_EVENT, self.DISASTER_RATE_PER_YEAR)

        if self.config.get("crisis_economica_enabled", False):
            self._schedule_next_global_event(EventType.CRISIS_EVENT, self.CRISIS_RATE_PER_YEAR)
        if self.config.get("baby_boom_enabled", False):
            self._schedule_next_global_event(EventType.BABY_BOOM_EVENT, self.BABY_BOOM_RATE_PER_YEAR)
        if self.config.get("cura_enfermedades_enabled", False):
            self._schedule_next_global_event(EventType.CURE_EVENT, self.CURE_RATE_PER_YEAR)
        if self.config.get("accidentes_masivo", False):
            self._schedule_next_global_event(EventType.ACCIDENT_EVENT, self.ACCIDENT_RATE_PER_YEAR)

    def _schedule_next_global_event(self, event_type: EventType, rate_per_year: float) -> None:
        """Agenda el siguiente evento global de un tipo."""
        if rate_per_year <= 0:
            return
        self.schedule_event(
            self.current_time_years + random.expovariate(rate_per_year),
            event_type,
        )

    def run(self, duration_years: float) -> None:
        """Ejecuta la simulacion en tiempo continuo durante duration_years."""
        self.build_initial_schedule()
        end_time = self.current_time_years + max(0.0, duration_years)

        while self.fel and self.current_time_years < end_time:
            event = heapq.heappop(self.fel)
            if event.time > end_time:
                heapq.heappush(self.fel, event)
                break

            delta_time = event.time - self.current_time_years
            self.age_population(delta_time)
            self.current_time_years = event.time
            self.process_gestations()
            self.dispatch_event(event)

        if self.current_time_years < end_time:
            self.age_population(end_time - self.current_time_years)
            self.current_time_years = end_time
            self.process_gestations()

        self.current_month = self.current_time_years * 12.0

    def age_population(self, delta_years: float) -> None:
        """Envejece vivos y fetos segun el salto de tiempo."""
        if delta_years <= 0.0:
            return

        if self.economic_crisis_active_years > 0:
            self.economic_crisis_active_years = max(0.0, self.economic_crisis_active_years - delta_years)
        if self.baby_boom_active_years > 0:
            self.baby_boom_active_years = max(0.0, self.baby_boom_active_years - delta_years)

        for person_id in list(self.alive_ids):
            person = self.people[person_id]
            person.age_years += delta_years
            if person.grief_time_remaining_years > 0:
                person.grief_time_remaining_years = max(0.0, person.grief_time_remaining_years - delta_years)

        for record in self.gestation_group.values():
            fetus = self.people[record.fetus_id]
            fetus.age_years += delta_years

    def process_gestations(self) -> None:
        """Convierte fetos con edad >= 0 en nacimientos efectivos."""
        ready = [fid for fid, record in self.gestation_group.items() if self.people[fid].age_years >= 0.0]
        for fetus_id in ready:
            self._finalize_birth(fetus_id)

    def dispatch_event(self, event: Event) -> None:
        """Despacha un evento segun su tipo."""
        if event.event_type == EventType.DEATH_EVENT:
            self._handle_death_event(event)
        elif event.event_type == EventType.PARTNER_DESIRE_EVENT:
            self._handle_partner_desire_event(event)
        elif event.event_type == EventType.COUPLE_FORMATION_EVENT:
            self._handle_couple_formation_event(event)
        elif event.event_type == EventType.CHILD_DESIRE_EVENT:
            self._handle_child_desire_event(event)
        elif event.event_type == EventType.PREGNANCY_ATTEMPT_EVENT:
            self._handle_pregnancy_attempt_event(event)
        elif event.event_type == EventType.PREGNANCY_START_EVENT:
            self._handle_pregnancy_start_event(event)
        elif event.event_type == EventType.BIRTH_EVENT:
            self._handle_birth_event(event)
        elif event.event_type == EventType.BREAKUP_EVENT:
            self._handle_breakup_event(event)
        elif event.event_type == EventType.EPIDEMIC_EVENT:
            self._handle_epidemic_event(event)
        elif event.event_type == EventType.WAR_EVENT:
            self._handle_war_event(event)
        elif event.event_type == EventType.DISASTER_EVENT:
            self._handle_disaster_event(event)
        elif event.event_type == EventType.CURE_EVENT:
            self._handle_cure_event(event)
        elif event.event_type == EventType.CRISIS_EVENT:
            self._handle_crisis_event(event)
        elif event.event_type == EventType.BABY_BOOM_EVENT:
            self._handle_baby_boom_event(event)
        elif event.event_type == EventType.ACCIDENT_EVENT:
            self._handle_accident_event(event)

    def _handle_death_event(self, event: Event) -> None:
        """Chequea mortalidad de una persona y reprograma la proxima."""
        if event.target_id is None:
            return
        person = self.people.get(event.target_id)
        if not person or not person.is_alive:
            return

        if person.age_years > 125:
            self._kill_person(person, "Muere de vejez extrema")
            return

        if person.age_years <= 12:
            annual_prob = 0.0025
        elif person.age_years <= 45:
            annual_prob = 0.0010 if person.sex == "M" else 0.0015
        elif person.age_years <= 76:
            annual_prob = 0.0030 if person.sex == "M" else 0.0035
        else:
            annual_prob = 0.070 if person.sex == "M" else 0.065

        if self.disease_cure_active:
            annual_prob *= 0.60
        if self.economic_crisis_active_years > 0:
            annual_prob *= 0.5
        if self.baby_boom_active_years > 0:
            annual_prob *= 1.5

        delta_years = self.current_time_years - person.last_mortality_check_time
        monthly_prob = self._annual_to_monthly_prob(annual_prob)
        period_prob = self._period_prob(monthly_prob, delta_years)
        person.last_mortality_check_time = self.current_time_years
        if evaluate_probability(period_prob):
            self._kill_person(person, "Muere por causas naturales")
            return

        self.schedule_event(
            self.current_time_years + random.expovariate(self.DEATH_CHECK_RATE_PER_YEAR),
            EventType.DEATH_EVENT,
            target_id=person.id,
        )

    def _handle_partner_desire_event(self, event: Event) -> None:
        """Actualiza el deseo de pareja de una persona."""
        if event.target_id is None:
            return
        person = self.people.get(event.target_id)
        if not person or not person.is_alive:
            return

        person.last_partner_desire_check_time = self.current_time_years
        if self._wants_partner(person):
            person.wants_partner = True
            self._update_ready_set(person)
        else:
            person.wants_partner = False
            self._update_ready_set(person)

        self.schedule_event(
            self.current_time_years + random.expovariate(self.PARTNER_DESIRE_RATE_PER_YEAR),
            EventType.PARTNER_DESIRE_EVENT,
            target_id=person.id,
        )

    def _handle_child_desire_event(self, event: Event) -> None:
        """Actualiza el deseo de hijos de una persona."""
        if event.target_id is None:
            return
        person = self.people.get(event.target_id)
        if not person or not person.is_alive:
            return

        person.wants_children = person.children_count < person.desired_children
        person.last_child_desire_check_time = self.current_time_years
        self.schedule_event(
            self.current_time_years + random.expovariate(self.CHILD_DESIRE_RATE_PER_YEAR),
            EventType.CHILD_DESIRE_EVENT,
            target_id=person.id,
        )

    def _handle_couple_formation_event(self, event: Event) -> None:
        """Intenta formar una nueva pareja y reprograma el evento global."""
        if self.ready_males and self.ready_females:
            male_id = random.choice(tuple(self.ready_males))
            female_id = random.choice(tuple(self.ready_females))
            male = self.people[male_id]
            female = self.people[female_id]

            base_prob = self._match_probability(male, female)
            noise = get_uniform(0.8, 1.2)
            prob = min(1.0, base_prob * noise)
            if evaluate_probability(prob):
                self._form_couple(male, female)

        self.schedule_event(
            self.current_time_years + random.expovariate(self.COUPLE_FORMATION_RATE_PER_YEAR),
            EventType.COUPLE_FORMATION_EVENT,
        )

    def _handle_pregnancy_attempt_event(self, event: Event) -> None:
        """Procesa un intento de embarazo en una pareja."""
        pair_key_raw = event.payload.get("pair_key")
        if not isinstance(pair_key_raw, str):
            return

        pair_key = tuple(int(x) for x in pair_key_raw.split("-"))
        relationship = self.relationships.get(pair_key)
        if relationship is None:
            return

        partner_a = self.people.get(relationship.partner_a)
        partner_b = self.people.get(relationship.partner_b)
        if not partner_a or not partner_b:
            return
        if not partner_a.is_alive or not partner_b.is_alive:
            return

        if partner_a.sex == "F":
            female = partner_a
            male = partner_b
        else:
            female = partner_b
            male = partner_a

        if female.is_pregnant:
            self._reschedule_pregnancy_attempt(relationship)
            return
        if not (female.wants_children and male.wants_children):
            self._reschedule_pregnancy_attempt(relationship)
            return

        age = female.age_years
        if age < 12:
            annual_prob = 0.0
        elif age <= 15:
            annual_prob = 0.20
        elif age <= 21:
            annual_prob = 0.45
        elif age <= 35:
            annual_prob = 0.80
        elif age <= 45:
            annual_prob = 0.40
        elif age <= 60:
            annual_prob = 0.20
        else:
            annual_prob = 0.05

        delta_years = self.current_time_years - relationship.last_pregnancy_attempt_time
        monthly_prob = self._annual_to_monthly_prob(annual_prob)
        period_prob = self._period_prob(monthly_prob, delta_years)
        relationship.last_pregnancy_attempt_time = self.current_time_years
        if evaluate_probability(period_prob):
            self.schedule_event(
                self.current_time_years,
                EventType.PREGNANCY_START_EVENT,
                target_id=female.id,
                payload={"father_id": male.id},
            )

        self._reschedule_pregnancy_attempt(relationship)

    def _reschedule_pregnancy_attempt(self, relationship: RelationshipRecord) -> None:
        """Reagenda un nuevo intento de embarazo."""
        pair_key = self._pair_key(relationship.partner_a, relationship.partner_b)
        self.schedule_event(
            self.current_time_years + random.expovariate(self.PREGNANCY_ATTEMPT_RATE_PER_YEAR),
            EventType.PREGNANCY_ATTEMPT_EVENT,
            payload={"pair_key": f"{pair_key[0]}-{pair_key[1]}"},
        )

    def _handle_pregnancy_start_event(self, event: Event) -> None:
        """Crea fetos con edad negativa y agenda nacimientos."""
        if event.target_id is None:
            return
        mother = self.people.get(event.target_id)
        if not mother or not mother.is_alive:
            return
        if mother.is_pregnant:
            return

        father_id = event.payload.get("father_id")
        if not isinstance(father_id, int):
            father_id = None

        r = random.random()
        if r < 0.686:
            num_babies = 1
        elif r < 0.862:
            num_babies = 2
        elif r < 0.941:
            num_babies = 3
        elif r < 0.980:
            num_babies = 4
        else:
            num_babies = 5

        mother.is_pregnant = True

        for _ in range(num_babies):
            sex = "M" if evaluate_probability(0.5) else "F"
            gestation_years = random.uniform(7, 10) / 12.0
            fetus = Person(sex, -gestation_years)
            fetus.desired_children = self._get_desired_children()
            self.people[fetus.id] = fetus
            self.gestation_group[fetus.id] = GestationRecord(
                fetus_id=fetus.id,
                mother_id=mother.id,
                father_id=father_id,
                due_time=self.current_time_years + gestation_years,
            )
            self.mother_gestations.setdefault(mother.id, set()).add(fetus.id)
            self.schedule_event(
                self.current_time_years + gestation_years,
                EventType.BIRTH_EVENT,
                target_id=fetus.id,
            )

    def _handle_birth_event(self, event: Event) -> None:
        """Procesa el nacimiento real de un feto."""
        if event.target_id is None:
            return
        if event.target_id not in self.gestation_group:
            return
        self._finalize_birth(event.target_id)

    def _finalize_birth(self, fetus_id: int) -> None:
        """Mueve un feto a poblacion activa y limpia su gestacion."""
        record = self.gestation_group.pop(fetus_id, None)
        if record is None:
            return
        fetus = self.people[fetus_id]
        fetus.age_years = max(0.0, fetus.age_years)
        fetus.is_alive = True
        self._add_person_to_population(fetus)
        self.total_births += 1

        mother = self.people.get(record.mother_id)
        if mother and mother.is_alive:
            mother.children_count += 1
        if record.father_id is not None:
            father = self.people.get(record.father_id)
            if father and father.is_alive:
                father.children_count += 1

        if mother and mother.id in self.mother_gestations:
            self.mother_gestations[mother.id].discard(fetus_id)
            if not self.mother_gestations[mother.id]:
                mother.is_pregnant = False
                self.mother_gestations.pop(mother.id, None)

        self._log_event(f"👶 Nacimiento: {fetus} llega al mundo.")

        self.schedule_event(
            self.current_time_years + random.expovariate(self.DEATH_CHECK_RATE_PER_YEAR),
            EventType.DEATH_EVENT,
            target_id=fetus.id,
        )
        self.schedule_event(
            self.current_time_years + random.expovariate(self.PARTNER_DESIRE_RATE_PER_YEAR),
            EventType.PARTNER_DESIRE_EVENT,
            target_id=fetus.id,
        )
        self.schedule_event(
            self.current_time_years + random.expovariate(self.CHILD_DESIRE_RATE_PER_YEAR),
            EventType.CHILD_DESIRE_EVENT,
            target_id=fetus.id,
        )

    def _handle_breakup_event(self, event: Event) -> None:
        """Evalua y ejecuta la ruptura de una pareja."""
        pair_key_raw = event.payload.get("pair_key")
        if not isinstance(pair_key_raw, str):
            return
        pair_key = tuple(int(x) for x in pair_key_raw.split("-"))
        relationship = self.relationships.get(pair_key)
        if relationship is None:
            return

        partner_a = self.people.get(relationship.partner_a)
        partner_b = self.people.get(relationship.partner_b)
        if not partner_a or not partner_b:
            return
        if not partner_a.is_alive or not partner_b.is_alive:
            return

        monthly_breakup_prob = self._annual_to_monthly_prob(0.20)
        delta_years = self.current_time_years - relationship.last_breakup_check_time
        period_prob = self._period_prob(monthly_breakup_prob, delta_years)
        relationship.last_breakup_check_time = self.current_time_years
        if evaluate_probability(period_prob):
            self._dissolve_relationship(pair_key, "💔 Ruptura de pareja")
            return

        self.schedule_event(
            self.current_time_years + random.expovariate(self.BREAKUP_CHECK_RATE_PER_YEAR),
            EventType.BREAKUP_EVENT,
            payload={"pair_key": f"{pair_key[0]}-{pair_key[1]}"},
        )

    def _handle_epidemic_event(self, event: Event) -> None:
        """Evento global de epidemia."""
        if not self.alive_ids:
            return
        self._log_event("⚠️ ¡ALERTA! Ha estallado una Epidemia Global mortal. Los más vulnerables corren peligro.")
        for person_id in list(self.alive_ids):
            person = self.people[person_id]
            if person.age_years < 3 or person.age_years > 65:
                if evaluate_probability(0.15):
                    self._kill_person(person, "Muere por la Epidemia")
            else:
                if evaluate_probability(0.02):
                    self._kill_person(person, "Muere por la Epidemia")

        self._schedule_next_global_event(EventType.EPIDEMIC_EVENT, self.EPIDEMIC_RATE_PER_YEAR)

    def _handle_war_event(self, event: Event) -> None:
        """Evento global de guerra."""
        if not self.alive_ids:
            return
        self._log_event("⚔️ ¡GUERRA! Ha estallado un conflicto armado. La población masculina es reclutada.")
        for person_id in list(self.alive_ids):
            person = self.people[person_id]
            if person.sex == "M" and 18 <= person.age_years <= 45:
                if evaluate_probability(0.20):
                    self._kill_person(person, "Soldado caído en combate:")
            else:
                if evaluate_probability(0.01):
                    self._kill_person(person, "Baja civil en la guerra:")

        self._schedule_next_global_event(EventType.WAR_EVENT, self.WAR_RATE_PER_YEAR)

    def _handle_disaster_event(self, event: Event) -> None:
        """Evento global de desastre natural."""
        if not self.alive_ids:
            return
        self._log_event("🌪️ ¡DESASTRE NATURAL! Un terremoto devastador ha destruido viviendas.")
        for person_id in list(self.alive_ids):
            person = self.people[person_id]
            if evaluate_probability(0.03):
                self._kill_person(person, "Fallecido en el desastre natural")

        self._schedule_next_global_event(EventType.DISASTER_EVENT, self.DISASTER_RATE_PER_YEAR)

    def _handle_cure_event(self, event: Event) -> None:
        """Evento global que activa la cura de enfermedades."""
        if not self.disease_cure_active:
            self.disease_cure_active = True
            self._log_event("🧬 ¡AVANCE MÉDICO! Se ha descubierto una cura universal para enfermedades graves.")

        self._schedule_next_global_event(EventType.CURE_EVENT, self.CURE_RATE_PER_YEAR)

    def _handle_crisis_event(self, event: Event) -> None:
        """Evento global de crisis economica."""
        if self.economic_crisis_active_years <= 0:
            duracion_meses = random.randint(12, 48)
            self.economic_crisis_active_years = duracion_meses / 12.0
            self._log_event(f"📉 CRISIS ECONÓMICA. Se avecinan tiempos difíciles por {duracion_meses} meses.")

        self._schedule_next_global_event(EventType.CRISIS_EVENT, self.CRISIS_RATE_PER_YEAR)

    def _handle_baby_boom_event(self, event: Event) -> None:
        """Evento global de baby boom."""
        if self.baby_boom_active_years <= 0:
            duracion_meses = random.randint(24, 60)
            self.baby_boom_active_years = duracion_meses / 12.0
            self._log_event(f"🎉 ÉPOCA DORADA. Baby Boom activo por {duracion_meses} meses.")

        self._schedule_next_global_event(EventType.BABY_BOOM_EVENT, self.BABY_BOOM_RATE_PER_YEAR)

    def _handle_accident_event(self, event: Event) -> None:
        """Evento global de accidentes aleatorios."""
        if not self.alive_ids:
            return
        victim_count = min(len(self.alive_ids), random.randint(1, 5))
        victims = random.sample(list(self.alive_ids), victim_count)
        for person_id in victims:
            person = self.people[person_id]
            self._kill_person(person, "Muere en un trágico accidente súbito")

        self._schedule_next_global_event(EventType.ACCIDENT_EVENT, self.ACCIDENT_RATE_PER_YEAR)

    def _form_couple(self, male: Person, female: Person) -> None:
        """Crea una relacion entre dos personas."""
        male.partner_id = female.id
        female.partner_id = male.id
        male.wants_partner = False
        female.wants_partner = False
        self._update_ready_set(male)
        self._update_ready_set(female)

        pair_key = self._pair_key(male.id, female.id)
        self.relationships[pair_key] = RelationshipRecord(
            partner_a=male.id,
            partner_b=female.id,
            last_breakup_check_time=self.current_time_years,
            last_pregnancy_attempt_time=self.current_time_years,
        )

        self._log_event(f"❤️ Nueva pareja formada: {male} y {female}")

        self.schedule_event(
            self.current_time_years + random.expovariate(self.BREAKUP_CHECK_RATE_PER_YEAR),
            EventType.BREAKUP_EVENT,
            payload={"pair_key": f"{pair_key[0]}-{pair_key[1]}"},
        )
        self._reschedule_pregnancy_attempt(self.relationships[pair_key])

    def _dissolve_relationship(self, pair_key: Tuple[int, int], reason: str) -> None:
        """Rompe una pareja y aplica duelo."""
        relationship = self.relationships.pop(pair_key, None)
        if relationship is None:
            return

        partner_a = self.people.get(relationship.partner_a)
        partner_b = self.people.get(relationship.partner_b)
        if partner_a and partner_b:
            partner_a.partner_id = None
            partner_b.partner_id = None
            partner_a.grief_time_remaining_years = self._get_grief_time_years(partner_a.age_years)
            partner_b.grief_time_remaining_years = self._get_grief_time_years(partner_b.age_years)
            self._update_ready_set(partner_a)
            self._update_ready_set(partner_b)
            self._log_event(f"{reason} entre {partner_a} y {partner_b}")

    def _update_ready_set(self, person: Person) -> None:
        """Mantiene sets de solteros listos para emparejar."""
        if person.partner_id is not None or not person.wants_partner:
            self.ready_males.discard(person.id)
            self.ready_females.discard(person.id)
            return
        if person.sex == "M":
            self.ready_males.add(person.id)
        else:
            self.ready_females.add(person.id)

    def _kill_person(self, person: Person, reason: str) -> None:
        """Mueve una persona a muertos y limpia su estado social."""
        if not person.is_alive:
            return
        person.is_alive = False
        self.total_deaths += 1
        self.alive_ids.discard(person.id)
        self.dead_ids.add(person.id)
        self.ready_males.discard(person.id)
        self.ready_females.discard(person.id)
        if person.sex == "M":
            self.male_group.discard(person.id)
        else:
            self.female_group.discard(person.id)

        if person.partner_id is not None:
            pair_key = self._pair_key(person.id, person.partner_id)
            self._dissolve_relationship(pair_key, "💀 Pareja termina por fallecimiento")

        self._log_event(f"💀 {reason} {person}")

    def tick(self) -> None:
        """Avanza exactamente un mes de simulacion (1/12 de año)."""
        self.run(1.0 / 12.0)
