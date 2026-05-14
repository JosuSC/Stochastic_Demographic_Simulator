import math
import random
from typing import Callable, List, Tuple
from models.person import Person
from utils.probability import get_uniform, evaluate_probability, get_exponential

class Simulator:
    """Motor central de eventos discretos con avance de tiempo por eventos."""
    def __init__(self, initial_females: int, initial_males: int):
        self.population: List[Person] = []
        self.current_month: float = 0.0
        self.total_births: int = 0
        self.total_deaths: int = 0

        # Historial de eventos
        self.current_year_logs: List[str] = []

        # Banderas de estado global
        self.disease_cure_active: bool = False
        self.economic_crisis_active: float = 0.0  # Meses restantes de crisis
        self.baby_boom_active: float = 0.0        # Meses restantes de auge demográfico

        # Sistema de flags para eventos globales
        self.config = {
            "crisis_economica_enabled": False,
            "baby_boom_enabled": False,
            "cura_enfermedades_enabled": False,
            "accidentes_masivo": False,
        }

        self._initialize_population(initial_females, 'F')
        self._initialize_population(initial_males, 'M')
        
    def _log_event(self, message: str) -> None:
        """Añade un evento al registro del año actual."""
        self.current_year_logs.append(message)

    def _annual_to_monthly_prob(self, annual_prob: float) -> float:
        """Convierte una probabilidad anual a mensual usando complementos."""
        if annual_prob >= 1.0: return 1.0
        if annual_prob <= 0.0: return 0.0
        return 1.0 - math.pow(1.0 - annual_prob, 1.0 / 12.0)

    def _period_prob(self, monthly_prob: float, delta_t: float) -> float:
        if monthly_prob <= 0.0: return 0.0
        if monthly_prob >= 1.0: return 1.0
        if delta_t <= 0.0: return 0.0
        return 1.0 - math.pow(1.0 - monthly_prob, delta_t)

    def _initialize_population(self, count: int, sex: str) -> None:
        """Crea la población inicial con edades uniformes entre 0 y 100 años."""
        for _ in range(count):
            age_months = get_uniform(0, 100) * 12
            p = Person(sex, age_months)
            
            # Asignar hijos deseados
            p.desired_children = self._get_desired_children()
            self.population.append(p)

    def _get_desired_children(self) -> int:
        # Suma original del problema: 0.6 + 0.75 + 0.35 + 0.2 + 0.1 + 0.05 = 2.05
        # Normalizamos para que la probabilidad total sea 1.0 (Probabilidad / 2.05)
        r = random.random()
        if r < 0.292: return 1         # 0.6 / 2.05
        elif r < 0.658: return 2       # + 0.75 / 2.05
        elif r < 0.829: return 3       # + 0.35 / 2.05
        elif r < 0.926: return 4       # + 0.2 / 2.05
        elif r < 0.975: return 5       # + 0.1 / 2.05
        else: return 6                 # + 0.05 / 2.05

    def get_alive_population(self) -> List[Person]:
        return [p for p in self.population if p.is_alive]

    def _check_death(self, p: Person) -> bool:
        """Chequea si muere usando distribución anual transformada de las reglas de vida real."""
        age = p.age_years
        if age > 125: 
            self._log_event(f"💀 Muere de vejez extrema {p}")
            return True
        
        # Asumimos que los valores dados eran porcentajes (ej. 0.25 = 0.25%).
        # Un 25% crudo anual causaría la extinción en una generación.
        if age <= 12: prob = 0.0025
        elif age <= 45: prob = 0.0010 if p.sex == 'M' else 0.0015
        elif age <= 76: prob = 0.0030 if p.sex == 'M' else 0.0035
        else: prob = 0.070 if p.sex == 'M' else 0.065
        
        # Si se descubrió la cura de enfermedades graves, la mortalidad general baja un 40%
        if self.disease_cure_active:
            prob *= 0.60
            
        # Modificador global por época económica o de auge
        if self.economic_crisis_active > 0:
            prob *= 0.5  # Baja la natalidad a la mitad en crisis
        if self.baby_boom_active > 0:
            prob *= 1.5  # Sube la natalidad en un 50%
            
        monthly_prob = self._annual_to_monthly_prob(prob)
        died = evaluate_probability(monthly_prob)
        if died:
            self._log_event(f"💀 Muere por causas naturales {p}")
        return died

    def _get_grief_time(self, age_years: float) -> float:
        if age_years <= 15: mean = 3
        elif age_years <= 21: mean = 6
        elif age_years <= 35: mean = 6
        elif age_years <= 45: mean = 12
        elif age_years <= 60: mean = 24
        else: mean = 48
        return max(1.0, get_exponential(mean))

    def _wants_partner(self, p: Person) -> bool:
        if p.partner is not None: return False
        if p.grief_time_remaining > 0: return False
        
        age = p.age_years
        if age < 12: prob = 0.0
        elif age <= 15: prob = 0.60
        elif age <= 21: prob = 0.65
        elif age <= 35: prob = 0.80
        elif age <= 45: prob = 0.60
        elif age <= 60: prob = 0.50
        else: prob = 0.20
        return evaluate_probability(prob)

    def _match_probability(self, p1: Person, p2: Person) -> float:
        # Prevenir emparejamientos bizarros entre un menor de edad y un adulto legal (aunque la dif sea < 20 años)
        is_p1_minor = p1.age_years < 18
        is_p2_minor = p2.age_years < 18
        if (is_p1_minor and not is_p2_minor) or (is_p2_minor and not is_p1_minor):
            return 0.0
            
        diff = abs(p1.age_years - p2.age_years)
        if diff <= 5: return 0.45
        elif diff <= 10: return 0.40
        elif diff <= 15: return 0.35
        elif diff <= 20: return 0.25
        else: return 0.15

    # EVENT SCHEDULER -------------------------------------------------
    def _build_events_by_type(
        self,
        funcion: Callable[[float], None],
        lam: float,
        duration: float,
    ) -> List[Tuple[float, Callable[[float], None]]]:
        if lam <= 0.0 or duration <= 0.0:
            return []
        t = 0.0
        agenda: List[Tuple[float, Callable[[float], None]]] = []
        while t < duration:
            dt = random.expovariate(lam)
            t += dt
            if t > duration:
                break
            agenda.append((t, funcion))
        return agenda

    def build_event_schedule(self, duration: float) -> List[Tuple[float, Callable[[float], None]]]:
        """Genera la agenda de eventos (tiempo, callback) ordenada cronológicamente."""
        agenda: List[Tuple[float, Callable[[float], None]]] = []

        # Tasas base (eventos por mes)
        tipos_eventos = [
            (self._event_deaths, 1.0),
            (self._event_breakups, 0.2),
            (self._event_matchmaking, 0.5),
            (self._event_pregnancies, 1.0),
            (self._event_epidemic, 0.01 / 12.0),
            (self._event_war, 0.005 / 12.0),
            (self._event_disaster, 0.01 / 12.0),
        ]

        for funcion, lam in tipos_eventos:
            agenda.extend(self._build_events_by_type(funcion, lam, duration))

        if self.config.get("crisis_economica_enabled", True):
            agenda.extend(self._build_events_by_type(self._event_crisis, 0.03 / 12.0, duration))
        if self.config.get("baby_boom_enabled", True):
            agenda.extend(self._build_events_by_type(self._event_baby_boom, 0.03 / 12.0, duration))
        if self.config.get("cura_enfermedades_enabled", True):
            agenda.extend(self._build_events_by_type(self._event_cure, 0.005 / 12.0, duration))
        if self.config.get("accidentes_aereos_enabled", True):
            agenda.extend(self._build_events_by_type(self._event_accidents, 0.05 / 12.0, duration))

        agenda.sort(key=lambda x: x[0])
        return agenda

    def _handle_birth(self, mother: Person) -> None:
        mother.is_pregnant = False
        mother.pregnant_months = 0
        
        # Suma original = 0.70 + 0.18 + 0.08 + 0.04 + 0.02 = 1.02. 
        # Normalizamos dividiendo por 1.02
        r = random.random()
        if r < 0.686: num_babies = 1       # 0.70 / 1.02
        elif r < 0.862: num_babies = 2     # + 0.18 / 1.02
        elif r < 0.941: num_babies = 3     # + 0.08 / 1.02
        elif r < 0.980: num_babies = 4     # + 0.04 / 1.02
        else: num_babies = 5
        
        for _ in range(num_babies):
            sex = 'M' if evaluate_probability(0.5) else 'F'
            baby = Person(sex, 0)
            baby.desired_children = self._get_desired_children()
            
            self.population.append(baby)
            self.total_births += 1
            mother.children_count += 1
            if mother.partner: mother.partner.children_count += 1
            
        self._log_event(f"👶 Nacimiento: {mother} acaba de dar a luz a {num_babies} bebé(s).")

    def _advance_time_state(self, delta_t: float) -> None:
        if delta_t <= 0.0:
            return

        if self.economic_crisis_active > 0:
            self.economic_crisis_active = max(0.0, self.economic_crisis_active - delta_t)
        if self.baby_boom_active > 0:
            self.baby_boom_active = max(0.0, self.baby_boom_active - delta_t)

        alive = self.get_alive_population()
        for p in alive:
            p.age_months += delta_t

            if p.grief_time_remaining > 0:
                p.grief_time_remaining = max(0.0, p.grief_time_remaining - delta_t)

            if p.is_pregnant:
                p.pregnant_months += delta_t
                if p.pregnant_months >= 9.0:
                    self._handle_birth(p)

    def _kill_person(self, p: Person, reason: str) -> None:
        p.is_alive = False
        self.total_deaths += 1
        self._log_event(f"💀 {reason} {p}")
        if p.partner:
            p.partner.partner = None
            p.partner.grief_time_remaining = self._get_grief_time(p.partner.age_years)

    # EVENT HANDLERS -------------------------------------------------
    def _event_deaths(self, delta_t: float) -> None:
        alive = self.get_alive_population()
        for p in alive:
            age = p.age_years
            if age > 125:
                self._kill_person(p, "Muere de vejez extrema")
                continue

            if age <= 12: prob = 0.0025
            elif age <= 45: prob = 0.0010 if p.sex == 'M' else 0.0015
            elif age <= 76: prob = 0.0030 if p.sex == 'M' else 0.0035
            else: prob = 0.070 if p.sex == 'M' else 0.065

            if self.disease_cure_active:
                prob *= 0.60
            if self.economic_crisis_active > 0:
                prob *= 0.5
            if self.baby_boom_active > 0:
                prob *= 1.5

            monthly_prob = self._annual_to_monthly_prob(prob)
            period_prob = self._period_prob(monthly_prob, delta_t)
            if evaluate_probability(period_prob):
                self._kill_person(p, "Muere por causas naturales")

    def _event_breakups(self, delta_t: float) -> None:
        monthly_breakup_prob = self._annual_to_monthly_prob(0.20)
        alive = self.get_alive_population()
        for p in alive:
            if p.partner and p.sex == 'M':
                if evaluate_probability(monthly_breakup_prob):
                    partner = p.partner
                    self._log_event(f"💔 Ruptura de pareja entre {p} y {partner}")
                    p.partner = None
                    partner.partner = None
                    p.grief_time_remaining = self._get_grief_time(p.age_years)
                    partner.grief_time_remaining = self._get_grief_time(partner.age_years)

    def _event_matchmaking(self, delta_t: float) -> None:
        alive = self.get_alive_population()
        singles_m = [p for p in alive if p.sex == 'M' and self._wants_partner(p)]
        singles_f = [p for p in alive if p.sex == 'F' and self._wants_partner(p)]

        random.shuffle(singles_m)
        random.shuffle(singles_f)

        limit = min(len(singles_m), len(singles_f))
        for i in range(limit):
            m = singles_m[i]
            f = singles_f[i]
            base_prob = self._match_probability(m, f)
            noise = get_uniform(0.8, 1.2)
            prob = min(1.0, base_prob * noise)
            if evaluate_probability(prob):
                m.partner = f
                f.partner = m
                self._log_event(f"❤️ Nueva pareja formada: {m} y {f}")

    def _event_pregnancies(self, delta_t: float) -> None:
        alive = self.get_alive_population()
        for p in alive:
            if p.sex != 'F' or p.is_pregnant:
                continue
            if p.partner and p.children_count < p.desired_children:
                age = p.age_years
                if age < 12: prob = 0.0
                elif age <= 15: prob = 0.20
                elif age <= 21: prob = 0.45
                elif age <= 35: prob = 0.80
                elif age <= 45: prob = 0.40
                elif age <= 60: prob = 0.20
                else: prob = 0.05

                monthly_preg_prob = self._annual_to_monthly_prob(prob)
                if evaluate_probability(monthly_preg_prob):
                    p.is_pregnant = True
                    p.pregnant_months = 0.0

    # EVENTOS GLOBALES ----------------------------------------------
    def _event_epidemic(self, delta_t: float) -> None:
        alive = self.get_alive_population()
        if not alive:
            return
        self._log_event("⚠️ ¡ALERTA! Ha estallado una Epidemia Global mortal. Los más vulnerables corren peligro.")
        for p in alive:
            if p.age_years < 3 or p.age_years > 65:
                if evaluate_probability(0.15):
                    self._kill_person(p, "Muere por la Epidemia")
            else:
                if evaluate_probability(0.02):
                    self._kill_person(p, "Muere por la Epidemia")

    def _event_war(self, delta_t: float) -> None:
        alive = self.get_alive_population()
        if not alive:
            return
        self._log_event("⚔️ ¡GUERRA! Ha estallado un conflicto armado. La población masculina es reclutada.")
        for p in alive:
            if p.sex == 'M' and 18 <= p.age_years <= 45:
                if evaluate_probability(0.20):
                    self._kill_person(p, "Soldado caído en combate:")
            else:
                if evaluate_probability(0.01):
                    self._kill_person(p, "Baja civil en la guerra:")

    def _event_disaster(self, delta_t: float) -> None:
        alive = self.get_alive_population()
        if not alive:
            return
        self._log_event("🌪️ ¡DESASTRE NATURAL! Un terremoto devastador ha destruido viviendas.")
        for p in alive:
            if evaluate_probability(0.03):
                self._kill_person(p, "Fallecido en el desastre natural")

    def _event_cure(self, delta_t: float) -> None:
        if not self.disease_cure_active:
            self.disease_cure_active = True
            self._log_event("🧬 ¡AVANCE MÉDICO! Se ha descubierto una cura universal para enfermedades graves.")

    def _event_accidents(self, delta_t: float) -> None:
        alive = self.get_alive_population()
        if not alive:
            return
        accident_victims = random.sample(alive, min(len(alive), random.randint(1, 5)))
        for v in accident_victims:
            self._kill_person(v, "Muere en un trágico accidente súbito")

    def _event_crisis(self, delta_t: float) -> None:
        if self.economic_crisis_active <= 0:
            duracion = random.randint(12, 48)
            self.economic_crisis_active = float(duracion)
            self._log_event(f"📉 CRISIS ECONÓMICA. Se avecinan tiempos difíciles por {duracion} meses.")

    def _event_baby_boom(self, delta_t: float) -> None:
        if self.baby_boom_active <= 0:
            duracion = random.randint(24, 60)
            self.baby_boom_active = float(duracion)
            self._log_event(f"🎉 ÉPOCA DORADA. Baby Boom activo por {duracion} meses.")

    # API PRINCIPAL --------------------------------------------------
    def run(self, duration: float) -> None:
        """Ejecuta la simulación durante `duration` meses usando agenda de eventos."""
        agenda_eventos = self.build_event_schedule(duration)
        last_time = 0.0

        for t, funcion in agenda_eventos:
            delta_t = t - last_time
            self._advance_time_state(delta_t)
            funcion(delta_t)
            last_time = t

        if last_time < duration:
            self._advance_time_state(duration - last_time)

        self.current_month += duration

    def tick(self) -> None:
        self.run(1.0)
