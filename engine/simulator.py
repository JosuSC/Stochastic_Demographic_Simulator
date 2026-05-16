# The `Simulator` class represents a discrete event simulation engine with various event types,
# population management, and relationship dynamics.
from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from models.person import Person
from utils.probability import get_uniform, evaluate_probability, get_exponential
from utils.tables_of_probabilities import tables, ProbabilityTable


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
    """Motor DES con FEL y tiempo continuo en años.

    El motor utiliza verificación periódica de eventos con probabilidades
    ajustadas al tiempo real transcurrido desde la última verificación.

    Formula clave: Si P_anual es la probabilidad anual de un evento,
    la probabilidad de que ocurra en un período de dt años es:
        P(dt) = 1 - (1 - P_anual)^dt
    """

    # Tasa de verificación de muerte: ~12 veces por año (cada mes)
    DEATH_CHECK_RATE_PER_YEAR = 12.0
    # Tasa de verificación de deseo de pareja: ~4 veces por año (cada trimestre)
    PARTNER_DESIRE_RATE_PER_YEAR = 4.0
    # Tasa de verificación de deseo de hijos: ~4 veces por año
    CHILD_DESIRE_RATE_PER_YEAR = 4.0
    # Tasa base de intentos de formación de pareja (se ajusta dinámicamente)
    COUPLE_FORMATION_BASE_RATE_PER_YEAR = 24.0
    # Tasa de intentos de embarazo por pareja: ~12 veces por año (cada mes)
    PREGNANCY_ATTEMPT_RATE_PER_YEAR = 12.0
    # Tasa de verificación de ruptura: ~6 veces por año (cada 2 meses)
    BREAKUP_CHECK_RATE_PER_YEAR = 6.0

    def __init__(self, initial_females: int, initial_males: int):
        """Arma el estado inicial del simulador y crea la poblacion base."""
        self.current_time_years: float = 0.0
        self.current_month: float = 0.0
        self.total_births: int = 0
        self.total_deaths: int = 0

        self.people: Dict[int, Person] = {}
        self.alive_ids: Set[int] = set()
        self.dead_ids: Set[int] = set()
        self.population_alive: List[Person] = []
        self.population_dead: List[Person] = []
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

        # Estadísticas por año para reportes
        self.yearly_stats: List[Dict] = []

        self._initialize_population(initial_females, "F")
        self._initialize_population(initial_males, "M")

    def _log_event(self, message: str) -> None:
        """Añade un evento al registro del año actual."""
        self.current_year_logs.append(message)

    def _initialize_population(self, count: int, sex: str) -> None:
        """Crea la poblacion inicial con edades uniformes entre 0 y 100 años (U(0,100))."""
        for _ in range(count):
            age_years = get_uniform(0, 100)
            person = Person(sex, age_years)
            person.desired_children = tables.desired_children_distribution.sample()
            # Inicializar tiempo de último chequeo de muerte al inicio de la simulación
            person.last_mortality_check_time = 0.0
            self._add_person_to_population(person)

    def _add_person_to_population(self, person: Person) -> None:
        """Registra una persona viva en los indices del simulador."""
        self.people[person.id] = person
        if person.id not in self.alive_ids:
            self.alive_ids.add(person.id)
            self.population_alive.append(person)
        if person.sex == "M":
            self.male_group.add(person.id)
        else:
            self.female_group.add(person.id)

    def _get_grief_time_years(self, age_years: float) -> float:
        """Devuelve el tiempo de duelo esperado en años.

        El tiempo de duelo sigue una distribución exponencial con media
        dada por la tabla 6 del problema (en meses).
        """
        mean_months = tables.grief_time.get_mean_months(age_years)
        # get_exponential(mean) devuelve un valor con esa media
        grief_months = get_exponential(mean_months)
        # Convertir a años, mínimo 1 mes
        return max(1.0 / 12.0, grief_months / 12.0)

    def _wants_partner(self, person: Person) -> bool:
        """Decide si alguien busca pareja en este momento.

        Usa la probabilidad anual de la tabla y la ajusta al período
        desde la última verificación.
        """
        if person.partner_id is not None:
            return False
        if person.grief_time_remaining_years > 0:
            return False

        age = person.age_years
        annual_prob = tables.partner_desire.get_annual_probability(age)

        # Ajustar probabilidad al tiempo desde el último chequeo
        delta = self.current_time_years - person.last_partner_desire_check_time
        if delta <= 0:
            return evaluate_probability(annual_prob)

        period_prob = ProbabilityTable.period_probability_from_annual(annual_prob, delta)
        return evaluate_probability(period_prob)

    def _match_probability(self, p1: Person, p2: Person) -> float:
        """Devuelve probabilidad de match segun diferencia de edad.

        Regla: menores de 18 no pueden emparejarse con mayores de 18.
        La probabilidad depende de la diferencia de edad (Tabla 5).
        """
        is_p1_minor = p1.age_years < 18
        is_p2_minor = p2.age_years < 18
        if (is_p1_minor and not is_p2_minor) or (is_p2_minor and not is_p1_minor):
            return 0.0

        diff = abs(p1.age_years - p2.age_years)
        return tables.couple_match.get_annual_probability(diff)

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
            # Chequeo de muerte
            self.schedule_event(
                self.current_time_years + random.expovariate(self.DEATH_CHECK_RATE_PER_YEAR),
                EventType.DEATH_EVENT,
                target_id=person.id,
            )
            # Chequeo de deseo de pareja
            self.schedule_event(
                self.current_time_years + random.expovariate(self.PARTNER_DESIRE_RATE_PER_YEAR),
                EventType.PARTNER_DESIRE_EVENT,
                target_id=person.id,
            )
            # Chequeo de deseo de hijos
            self.schedule_event(
                self.current_time_years + random.expovariate(self.CHILD_DESIRE_RATE_PER_YEAR),
                EventType.CHILD_DESIRE_EVENT,
                target_id=person.id,
            )

        # Evento global de formación de parejas
        self.schedule_event(
            self.current_time_years + random.expovariate(self._couple_formation_rate()),
            EventType.COUPLE_FORMATION_EVENT,
        )

    def _couple_formation_rate(self) -> float:
        """Calcula la tasa dinámica de formación de parejas.

        La tasa se escala según la cantidad de personas disponibles:
        más solteros = más intentos de emparejamiento.
        Mínimo 24 intentos/año, escalado por el número de listos.
        """
        n_ready = len(self.ready_males) + len(self.ready_females)
        if n_ready <= 0:
            return self.COUPLE_FORMATION_BASE_RATE_PER_YEAR
        # Al menos 2 intentos por persona lista por año, con un mínimo base
        return max(self.COUPLE_FORMATION_BASE_RATE_PER_YEAR, n_ready * 2.0)

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

    def _handle_death_event(self, event: Event) -> None:
        """Chequea mortalidad de una persona y reprograma la proxima.

        CORRECCIÓN PRINCIPAL:
        - Usa el tiempo REAL desde el último chequeo como delta
        - Aplica la fórmula correcta: P(período) = 1 - (1 - P_anual)^dt
        - No usa el ancho de la banda de edad como delta
        """
        if event.target_id is None:
            return
        person = self.people.get(event.target_id)
        if not person or not person.is_alive:
            return

        # Muerte por vejez extrema
        if person.age_years > 125:
            self._kill_person(person, "Muere de vejez extrema")
            return

        # Obtener probabilidad anual según edad y sexo
        if person.sex == "M":
            annual_prob = tables.death_male.get_annual_probability(person.age_years)
        else:
            annual_prob = tables.death_female.get_annual_probability(person.age_years)

        # Calcular tiempo real desde el último chequeo de mortalidad
        delta_years = self.current_time_years - person.last_mortality_check_time
        person.last_mortality_check_time = self.current_time_years

        # Probabilidad de morir en el período transcurrido:
        # P(dt) = 1 - (1 - P_anual)^dt
        if delta_years > 0 and annual_prob > 0:
            period_prob = 1.0 - math.pow(1.0 - annual_prob, delta_years)
            if evaluate_probability(period_prob):
                self._kill_person(person, "Muere por causas naturales")
                return

        # Reprogramar próximo chequeo de muerte
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

        # Solo las personas en edad fértil pueden desear hijos
        if person.age_years < 12:
            person.wants_children = False
        else:
            person.wants_children = person.children_count < person.desired_children

        person.last_child_desire_check_time = self.current_time_years
        self.schedule_event(
            self.current_time_years + random.expovariate(self.CHILD_DESIRE_RATE_PER_YEAR),
            EventType.CHILD_DESIRE_EVENT,
            target_id=person.id,
        )

    def _handle_couple_formation_event(self, event: Event) -> None:
        """Intenta formar una nueva pareja y reprograma el evento global.

        CORRECCIÓN: La tasa de formación es dinámica, escalada por el
        número de personas disponibles, para que haya suficientes
        intentos de emparejamiento.
        """
        # Intentar formar múltiples parejas por evento para mayor eficiencia
        attempts = max(1, min(len(self.ready_males), len(self.ready_females), 3))

        for _ in range(attempts):
            if not self.ready_males or not self.ready_females:
                break

            male_id = random.choice(tuple(self.ready_males))
            female_id = random.choice(tuple(self.ready_females))
            male = self.people[male_id]
            female = self.people[female_id]

            # Verificar que siguen vivos y disponibles
            if not male.is_alive or not female.is_alive:
                self.ready_males.discard(male_id)
                self.ready_females.discard(female_id)
                continue
            if male.partner_id is not None or female.partner_id is not None:
                self._update_ready_set(male)
                self._update_ready_set(female)
                continue

            base_prob = self._match_probability(male, female)
            # Pequeño ruido para variabilidad
            noise = get_uniform(0.85, 1.15)
            prob = min(1.0, base_prob * noise)
            if evaluate_probability(prob):
                self._form_couple(male, female)

        # Reprogramar con tasa dinámica
        self.schedule_event(
            self.current_time_years + random.expovariate(self._couple_formation_rate()),
            EventType.COUPLE_FORMATION_EVENT,
        )

    def _handle_pregnancy_attempt_event(self, event: Event) -> None:
        """Procesa un intento de embarazo en una pareja.

        CORRECCIÓN: Usa la fórmula correcta de probabilidad periódica
        con el tiempo real desde el último intento.
        """
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

        # Verificar condiciones previas
        if female.is_pregnant:
            self._reschedule_pregnancy_attempt(relationship)
            return
        if not (female.wants_children and male.wants_children):
            self._reschedule_pregnancy_attempt(relationship)
            return

        # Calcular probabilidad de embarazo
        age = female.age_years
        annual_prob = tables.pregnancy.get_annual_probability(age)

        # Tiempo desde el último intento
        delta_years = self.current_time_years - relationship.last_pregnancy_attempt_time
        relationship.last_pregnancy_attempt_time = self.current_time_years

        if delta_years > 0 and annual_prob > 0:
            # Probabilidad de quedar embarazada en el período transcurrido
            period_prob = 1.0 - math.pow(1.0 - annual_prob, delta_years)
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

        num_babies = tables.babies_distribution.sample()

        mother.is_pregnant = True

        for _ in range(num_babies):
            sex = "M" if evaluate_probability(0.5) else "F"
            gestation_years = random.uniform(7, 10) / 12.0
            # Feto con edad negativa (aún no nacido) y is_alive = False
            fetus = Person(sex, -gestation_years, is_alive=False)
            fetus.desired_children = tables.desired_children_distribution.sample()
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
        # CORRECCIÓN: Inicializar el tiempo de último chequeo de muerte
        # al momento actual, no a 0.0 (evita delta enorme en primer chequeo)
        fetus.last_mortality_check_time = self.current_time_years
        fetus.last_partner_desire_check_time = self.current_time_years
        fetus.last_child_desire_check_time = self.current_time_years

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

        # Agendar eventos para el recién nacido
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
        """Evalua y ejecuta la ruptura de una pareja.

        CORRECCIÓN: Usa la fórmula correcta de probabilidad periódica
        con el tiempo real desde el último chequeo.
        """
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

        # Probabilidad anual de ruptura (constante 0.20)
        annual_breakup = tables.breakup.get_annual_probability(0.0)

        # Tiempo real desde el último chequeo de ruptura
        delta_years = self.current_time_years - relationship.last_breakup_check_time
        relationship.last_breakup_check_time = self.current_time_years

        if delta_years > 0 and annual_breakup > 0:
            # Probabilidad de ruptura en el período transcurrido
            period_prob = 1.0 - math.pow(1.0 - annual_breakup, delta_years)
            if evaluate_probability(period_prob):
                self._dissolve_relationship(pair_key, "💔 Ruptura de pareja")
                return

        self.schedule_event(
            self.current_time_years + random.expovariate(self.BREAKUP_CHECK_RATE_PER_YEAR),
            EventType.BREAKUP_EVENT,
            payload={"pair_key": f"{pair_key[0]}-{pair_key[1]}"},
        )

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
        if person.partner_id is not None or not person.wants_partner or not person.is_alive:
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
        if person in self.population_alive:
            self.population_alive.remove(person)
        if person not in self.population_dead:
            self.population_dead.append(person)
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

    def get_statistics(self) -> Dict:
        """Devuelve estadísticas actuales de la simulación."""
        alive = [p for p in self.population_alive if p.is_alive]
        if not alive:
            return {
                "time_years": self.current_time_years,
                "population": 0,
                "males": 0,
                "females": 0,
                "avg_age": 0,
                "births": self.total_births,
                "deaths": self.total_deaths,
                "couples": len(self.relationships),
                "pregnant": sum(1 for p in alive if p.is_pregnant),
            }

        males = [p for p in alive if p.sex == "M"]
        females = [p for p in alive if p.sex == "F"]
        ages = [p.age_years for p in alive]

        return {
            "time_years": round(self.current_time_years, 2),
            "population": len(alive),
            "males": len(males),
            "females": len(females),
            "avg_age": round(sum(ages) / len(ages), 1) if ages else 0,
            "max_age": round(max(ages), 1) if ages else 0,
            "min_age": round(min(ages), 1) if ages else 0,
            "births": self.total_births,
            "deaths": self.total_deaths,
            "couples": len(self.relationships),
            "pregnant": sum(1 for p in alive if p.is_pregnant),
            "ready_males": len(self.ready_males),
            "ready_females": len(self.ready_females),
        }
