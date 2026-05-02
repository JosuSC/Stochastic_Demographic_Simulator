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
        
        self._initialize_population(initial_females, 'F')
        self._initialize_population(initial_males, 'M')

    def _initialize_population(self, count: int, sex: str) -> None:
        """Crea la población inicial con edades uniformes entre 0 y 100 años."""
        for _ in range(count):
            age_months = int(get_uniform(0, 100) * 12)
            p = Person(sex, age_months)
            
            # Asignar hijos deseados
            p.desired_children = self._get_desired_children()
            self.population.append(p)

    def _get_desired_children(self) -> int:
        r = random.random()
        if r < 0.6: return 1
        if r < 0.6 + 0.75 / 2.5: return 2 # Normalizado aprox
        return random.randint(1, 5)

    def get_alive_population(self) -> List[Person]:
        return [p for p in self.population if p.is_alive]

    def tick(self) -> None:
        """Avanza el reloj un mes y procesa todos los eventos para la población."""
        self.current_month += 1
        alive = self.get_alive_population()
        
        # Procesar envejecimiento y muertes
        for p in alive:
            p.age_one_month()
            if self._check_death(p):
                p.is_alive = False
                self.total_deaths += 1
                if p.partner:
                    p.partner.partner = None
                    p.partner.grief_time_remaining = self._get_grief_time(p.partner.age_years)
                    
        # ... Lógica simplificada de emparejamiento y nacimientos para acortar
        
    def run(self, months: int) -> None:
        """Ejecuta la simulación durante el número de meses especificado."""
        for _ in range(months):
            self.tick()

    def _check_death(self, p: Person) -> bool:
        return evaluate_probability(0.001) # Dummy prob

    def _get_grief_time(self, age_years: float) -> int:
        return int(get_exponential(12.0))
