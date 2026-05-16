from typing import Optional

class Person:
    """
    Representa un agente (persona) en la simulación demográfica.
    """
    _id_counter = 0

    def __init__(self, sex: str, age_years: float, is_alive: bool = True):
        """
        Inicializa una nueva persona.
        
        Args:
            sex: 'M' (Male) o 'F' (Female).
            age_years: Edad inicial en años (puede ser negativa para fetos).
            is_alive: Si la persona está viva. Los fetos nacen con is_alive=False
                      y se cambia a True en el nacimiento.
        """
        self.id: int = Person._id_counter
        Person._id_counter += 1
        
        self.sex: str = sex
        self.age_years: float = age_years
        self.is_alive: bool = is_alive
        self.partner_id: Optional[int] = None
        self.grief_time_remaining_years: float = 0.0
        
        self.desired_children: int = 0
        self.children_count: int = 0
        self.is_pregnant: bool = False
        self.wants_partner: bool = False
        self.wants_children: bool = False

        # Tiempos de último chequeo para cálculos de probabilidad periódica
        self.last_mortality_check_time: float = 0.0
        self.last_partner_desire_check_time: float = 0.0
        self.last_child_desire_check_time: float = 0.0
    
    @property
    def age_months(self) -> float:
        """Devuelve la edad en meses para reportes."""
        return self.age_years * 12.0
        
    def __str__(self) -> str:
        return f"Persona #{self.id} ({self.sex}, {int(self.age_years)} años)"
