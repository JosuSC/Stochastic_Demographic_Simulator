from typing import Optional, List

class Person:
    """
    Representa un agente (persona) en la simulación demográfica.
    """
    _id_counter = 0

    def __init__(self, sex: str, age_months: int):
        """
        Inicializa una nueva persona.
        
        Args:
            sex: 'M' (Male) o 'F' (Female).
            age_months: Edad inicial en meses.
        """
        self.id: int = Person._id_counter
        Person._id_counter += 1
        
        self.sex: str = sex
        self.age_months: int = age_months
        self.is_alive: bool = True
        self.partner: Optional['Person'] = None
        self.grief_time_remaining: int = 0
        
        self.desired_children: int = 0
        self.children_count: int = 0
        self.pregnant_months: int = 0
        self.is_pregnant: bool = False
    
    @property
    def age_years(self) -> float:
        """Devuelve la edad en años."""
        return self.age_months / 12.0
        
    def __str__(self) -> str:
        return f"Persona #{self.id} ({self.sex}, {int(self.age_years)} años)"
        
    def age_one_month(self) -> None:
        """Envejece a la persona por un mes y actualiza sus estados temporales."""
        self.age_months += 1
        if self.grief_time_remaining > 0:
            self.grief_time_remaining -= 1
        if self.is_pregnant:
            self.pregnant_months += 1
