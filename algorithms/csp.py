from __future__ import annotations
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from algorithms.problems_csp import DroneAssignmentCSP


def backtracking_search(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Basic backtracking search without optimizations.

    Tips:
    - An assignment is a dictionary mapping variables to values (e.g. {X1: Cell(1,2), X2: Cell(3,4)}).
    - Use csp.assign(var, value, assignment) to assign a value to a variable.
    - Use csp.unassign(var, assignment) to unassign a variable.
    - Use csp.is_consistent(var, value, assignment) to check if an assignment is consistent with the constraints.
    - Use csp.is_complete(assignment) to check if the assignment is complete (all variables assigned).
    - Use csp.get_unassigned_variables(assignment) to get a list of unassigned variables.
    - Use csp.domains[var] to get the list of possible values for a variable.
    - Use csp.get_neighbors(var) to get the list of variables that share a constraint with var.
    - Add logs to measure how good your implementation is (e.g. number of assignments, backtracks).

    You can find inspiration in the textbook's pseudocode:
    Artificial Intelligence: A Modern Approach (4th Edition) by Russell and Norvig, Chapter 5: Constraint Satisfaction Problems
    """
    assignment = {}
    return backtrack(csp, assignment)
    
def backtrack(csp, assignment):
    # Caso base
    if csp.is_complete(assignment):
        return assignment
    # Escoger una variable no asignada
    unassigned = csp.get_unassigned_variables(assignment)
    var = unassigned[0]
    # Probar valores del dominio
    for value in csp.domains[var]:
        # Revisar si es consistente
        if csp.is_consistent(var, value, assignment):
            # Asignar valor
            csp.assign(var, value, assignment)
            # Llamada recursiva
            result = backtrack(csp, assignment)
            if result is not None:
                return result
            # Backtracking
            csp.unassign(var, assignment)

    return None


def backtracking_fc(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking search with Forward Checking.

    Tips:
    - Forward checking: After assigning a value to a variable, eliminate inconsistent values from
      the domains of unassigned neighbors. If any neighbor's domain becomes empty, backtrack immediately.
    - Save domains before forward checking so you can restore them on backtrack.
    - Use csp.get_neighbors(var) to get variables that share constraints with var.
    - Use csp.is_consistent(neighbor, val, assignment) to check if a value is still consistent.
    - Forward checking reduces the search space by detecting failures earlier than basic backtracking.
    """
    assignment = {}
    return backtrack_fc(csp, assignment)

def backtrack_fc(csp, assignment):
    # Asignación está completa
    if csp.is_complete(assignment):
        return assignment
    # Escoger una variable no asignada
    var = csp.get_unassigned_variables(assignment)[0]
    for value in csp.domains[var]:
        if csp.is_consistent(var, value, assignment):
            # Asignar valor
            csp.assign(var, value, assignment)
            # Guardar copia de los dominios
            saved_domains = {v: list(csp.domains[v]) for v in csp.domains}
            # Forward checking
            fail = False
            for neighbor in csp.get_neighbors(var):
                if neighbor not in assignment:
                    new_domain = []
                    for val in csp.domains[neighbor]:
                        if csp.is_consistent(neighbor, val, assignment):
                            new_domain.append(val)
                    csp.domains[neighbor] = new_domain
                    # Si el dominio queda vacío, falla
                    if len(new_domain) == 0:
                        fail = True
                        break
            # Continúa búsqueda si no falló
            if not fail:
                result = backtrack_fc(csp, assignment)
                if result is not None:
                    return result
            # Restaurar dominios
            csp.domains = saved_domains
            # Backtracking
            csp.unassign(var, assignment)

    return None


def backtracking_ac3(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking search with AC-3 arc consistency.

    Tips:
    - AC-3 enforces arc consistency: for every pair of constrained variables (Xi, Xj), every value
      in Xi's domain must have at least one supporting value in Xj's domain.
    - Run AC-3 before starting backtracking to reduce domains globally.
    - After each assignment, run AC-3 on arcs involving the assigned variable's neighbors.
    - If AC-3 empties any domain, the current assignment is inconsistent - backtrack.
    - You can create helper functions such as:
      - a values_compatible function to check if two variable-value pairs are consistent with the constraints.
      - a revise function that removes unsupported values from one variable's domain.
      - an ac3 function that manages the queue of arcs to check and calls revise.
      - a backtrack function that integrates AC-3 into the search process.
    """
    assignment = {}
    # Guardar dominios originales
    saved_domains = {v: list(csp.domains[v]) for v in csp.domains}
    # Ejecutar AC3 antes de empezar
    if not ac3(csp):
        return None
    result = backtrack_ac3(csp, assignment)
    # Restaurar dominios originales
    csp.domains = saved_domains
    return result

def backtrack_ac3(csp, assignment):
    if csp.is_complete(assignment):
        return assignment
    var = csp.get_unassigned_variables(assignment)[0]
    for value in csp.domains[var]:
        if csp.is_consistent(var, value, assignment):
            csp.assign(var, value, assignment)
            # Guardar dominios
            saved_domains = {v: list(csp.domains[v]) for v in csp.domains}
            # Restringir dominio de la variable asignada
            csp.domains[var] = [value]
            # Ejecutar AC3 en vecinos
            queue = [(neighbor, var) for neighbor in csp.get_neighbors(var)]
            if ac3(csp, queue):
                result = backtrack_ac3(csp, assignment)
                if result is not None:
                    return result
            # Restaurar dominios
            csp.domains = saved_domains
            csp.unassign(var, assignment)
    return None

def ac3(csp, queue=None):
    if queue is None:
        queue = deque()
        for xi in csp.domains:
            for xj in csp.get_neighbors(xi):
                queue.append((xi, xj))
    else:
        queue = deque(queue)
    while queue:
        xi, xj = queue.popleft()
        if revise(csp, xi, xj):
            if len(csp.domains[xi]) == 0:
                return False
            for xk in csp.get_neighbors(xi):
                if xk != xj:
                    queue.append((xk, xi))
    return True


def revise(csp, xi, xj):
    revised = False
    for x in list(csp.domains[xi]):
        supported = False
        for y in csp.domains[xj]:
            temp_assignment = {xi: x, xj: y}
            if csp.is_consistent(xi, x, {xj: y}) and csp.is_consistent(xj, y, {xi: x}):
                supported = True
                break
        if not supported:
            csp.domains[xi].remove(x)
            revised = True
    return revised


def backtracking_mrv_lcv(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking with Forward Checking + MRV + LCV.

    Tips:
    - Combine the techniques from backtracking_fc, mrv_heuristic, and lcv_heuristic.
    - MRV (Minimum Remaining Values): Select the unassigned variable with the fewest legal values.
      Tie-break by degree: prefer the variable with the most unassigned neighbors.
    - LCV (Least Constraining Value): When ordering values for a variable, prefer
      values that rule out the fewest choices for neighboring variables.
    - Use csp.get_num_conflicts(var, value, assignment) to count how many values would be ruled out for neighbors if var=value is assigned.
    """
    assignment = {}
    return backtrack_mrv_lcv(csp, assignment)


def backtrack_mrv_lcv(csp, assignment):
    if csp.is_complete(assignment):
        return assignment
    # MRV
    unassigned = csp.get_unassigned_variables(assignment)
    var = min(unassigned, key=lambda v: len(csp.domains[v]))
    # LCV
    values = sorted(
        csp.domains[var],
        key=lambda val: csp.get_num_conflicts(var, val, assignment)
    )
    for value in values:
        if csp.is_consistent(var, value, assignment):
            csp.assign(var, value, assignment)
            # Guardar dominios
            saved_domains = {v: list(csp.domains[v]) for v in csp.domains}
            # Forward Checking
            fail = False
            for neighbor in csp.get_neighbors(var):
                if neighbor not in assignment:
                    new_domain = []
                    for val in csp.domains[neighbor]:
                        if csp.is_consistent(neighbor, val, assignment):
                            new_domain.append(val)
                    csp.domains[neighbor] = new_domain
                    if len(new_domain) == 0:
                        fail = True
                        break
            if not fail:
                result = backtrack_mrv_lcv(csp, assignment)
                if result is not None:
                    return result
            # Restaurar dominios
            csp.domains = saved_domains
            csp.unassign(var, assignment)
    return None
