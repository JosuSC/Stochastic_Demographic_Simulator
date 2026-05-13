import math
import random
from typing import List, Tuple
from models.person import Person
from utils.probability import get_uniform, evaluate_probability, get_exponential

class Simulator:
    """Motor central de eventos discretos que maneja el reloj y los agentes."""
    def __init__(self, initial_females: int, initial_males: int):
        self.population: List[Person] = []
        self.current_month: int = 0
        self.total_births: int = 0
        self.total_deaths: int = 0
        
        # Historial de eventos
        self.current_year_logs: List[str] = []
        
        # Banderas de estado global
        self.disease_cure_active: bool = False  
        self.economic_crisis_active: int = 0    # Meses restantes de crisis
        self.baby_boom_active: int = 0          # Meses restantes de auge demográfico
        
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

    def _initialize_population(self, count: int, sex: str) -> None:
        """Crea la población inicial con edades uniformes entre 0 y 100 años."""
        for _ in range(count):
            age_months = int(get_uniform(0, 100) * 12)
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

    def _get_grief_time(self, age_years: float) -> int:
        if age_years <= 15: mean = 3
        elif age_years <= 21: mean = 6
        elif age_years <= 35: mean = 6
        elif age_years <= 45: mean = 12
        elif age_years <= 60: mean = 24
        else: mean = 48
        return max(1, int(get_exponential(mean)))

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

    def _process_matchmaking(self, alive: List[Person]) -> None:
        """Empareja solteros eficientemente."""
        singles_m = [p for p in alive if p.sex == 'M' and self._wants_partner(p)]
        singles_f = [p for p in alive if p.sex == 'F' and self._wants_partner(p)]
        
        random.shuffle(singles_m)
        random.shuffle(singles_f)
        
        limit = min(len(singles_m), len(singles_f))
        for i in range(limit):
            m = singles_m[i]
            f = singles_f[i]
            prob = self._match_probability(m, f)
            if evaluate_probability(prob):
                m.partner = f
                f.partner = m
                self._log_event(f"❤️ Nueva pareja formada: {m} y {f}")

    def _process_breakups(self, alive: List[Person]) -> None:
        monthly_breakup_prob = self._annual_to_monthly_prob(0.20)
        for p in alive:
            if p.partner and p.sex == 'M': # Iterar solo por los machos evita procesar la pareja 2 veces.
                if evaluate_probability(monthly_breakup_prob):
                    partner = p.partner
                    self._log_event(f"💔 Ruptura de pareja entre {p} y {partner}")
                    p.partner = None
                    partner.partner = None
                    p.grief_time_remaining = self._get_grief_time(p.age_years)
                    partner.grief_time_remaining = self._get_grief_time(partner.age_years)

    def _process_pregnancies_and_births(self, alive: List[Person]) -> None:
        for p in alive:
            if p.sex != 'F': continue
            
            if p.is_pregnant:
                if p.pregnant_months >= 9:
                    self._handle_birth(p)
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
                    p.pregnant_months = 0

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

    def tick(self) -> None:
        self.current_month += 1
        
        # Decrementar contadores de estados globales
        if self.economic_crisis_active > 0: self.economic_crisis_active -= 1
        if self.baby_boom_active > 0: self.baby_boom_active -= 1
        
        alive = self.get_alive_population()
        
        # 1. Envejecer y muertes
        for p in alive:
            p.age_one_month()
            if self._check_death(p):
                p.is_alive = False
                self.total_deaths += 1
                if p.partner:
                    p.partner.partner = None
                    p.partner.grief_time_remaining = self._get_grief_time(p.partner.age_years)
                    
        alive = self.get_alive_population()
        
        # 2. Rupturas
        self._process_breakups(alive)
        
        # 3. Matchmaking
        self._process_matchmaking(alive)
        
        # 4. Embarazos
        self._process_pregnancies_and_births(alive)
        
        # 5. Evaluador de Eventos Globales de impacto (ocurren anualmente, por convención el mes 12)
        if self.current_month % 12 == 0:
            self._evaluate_global_events(self.get_alive_population())

    def _evaluate_global_events(self, alive: List[Person]) -> None:
        """Motor de azar para eventos globales que cambian el rumbo del mundo entero."""
        
        # 1. Epidemia Global (1% de probabilidad anual)
        if evaluate_probability(0.01):
            self._log_event("⚠️ ¡ALERTA! Ha estallado una Epidemia Global mortal. Los más vulnerables corren peligro.")
            for p in alive:
                # Más mortal para ancianos y bebés
                if p.age_years < 3 or p.age_years > 65:
                    if evaluate_probability(0.15): # 15% mueren
                        p.is_alive = False
                        self.total_deaths += 1
                        self._log_event(f"💀 Muere por la Epidemia {p}")
                        if p.partner: p.partner.partner = None
                else: # 2% en población sana
                    if evaluate_probability(0.02):
                        p.is_alive = False
                        self.total_deaths += 1
                        self._log_event(f"💀 Muere por la Epidemia {p}")
                        if p.partner: p.partner.partner = None

        # 2. Guerra Mundial/Civil (0.5% probabilidad)
        elif evaluate_probability(0.005):
            self._log_event("⚔️ ¡GUERRA! Ha estallado un conflicto armado. La población masculina es reclutada.")
            for p in alive:
                if p.sex == 'M' and 18 <= p.age_years <= 45:
                    if evaluate_probability(0.20): # 20% baja en el frente de batalla
                        p.is_alive = False
                        self.total_deaths += 1
                        self._log_event(f"💀 Soldado caído en combate: {p}")
                        if p.partner: p.partner.partner = None
                else: # Daño colateral
                    if evaluate_probability(0.01):
                        p.is_alive = False
                        self.total_deaths += 1
                        self._log_event(f"💀 Baja civil en la guerra: {p}")
                        if p.partner: p.partner.partner = None
                        
        # 3. Desastre Natural Terremoto / Tsunami (1% de probabilidad)
        elif evaluate_probability(0.01):
            self._log_event("🌪️ ¡DESASTRE NATURAL! Un terremoto devastador ha destruido viviendas.")
            for p in alive:
                if evaluate_probability(0.03): # Muerte al 3% aleatorio
                    p.is_alive = False
                    self.total_deaths += 1
                    self._log_event(f"💀 Fallecido en el desastre natural {p}")
                    if p.partner: p.partner.partner = None

        # 4. Avance Científico Médico (0.5%)
        if not self.disease_cure_active and evaluate_probability(0.005):
            self.disease_cure_active = True
            self._log_event("🧬 ¡AVANCE MÉDICO! Se ha descubierto una cura universal para enfermedades graves. La esperanza de vida sube drásticamente.")
            
        # 5. Accidentes Esporádicos (Aviones, Tránsito grave) (5% anual)
        if evaluate_probability(0.05):
            accident_victims = random.sample(alive, min(len(alive), random.randint(1, 5)))
            for v in accident_victims:
                v.is_alive = False
                self.total_deaths += 1
                self._log_event(f"💀 Muere en un trágico accidente súbito {v}")
                if v.partner: v.partner.partner = None
                
        # 6. Fluctuaciones Económicas y Sociales
        if self.economic_crisis_active <= 0 and evaluate_probability(0.03):
            duracion = random.randint(12, 48) # de 1 a 4 años
            self.economic_crisis_active = duracion
            self._log_event(f"📉 CRISIS ECONÓMICA. Se avecinan tiempos difíciles por los próximos {duracion//12} años. Baja la natalidad.")
            
        elif self.baby_boom_active <= 0 and evaluate_probability(0.03):
            duracion = random.randint(24, 60) # de 2 a 5 años
            self.baby_boom_active = duracion
            self._log_event(f"🎉 ¡ÉPOCA DORADA! Prosperidad económica desata un Baby Boom por los próximos {duracion//12} años.")

    def run(self, months: int) -> None:
        """Ejecuta la simulación durante el número de meses especificado."""
        for _ in range(months):
            self.tick()
