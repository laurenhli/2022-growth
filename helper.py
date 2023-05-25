import numpy as np
from typing import Optional

# A generation is (represented by) a list of cells.
# A trajectory is a list of generations.

# Useful classes

class Cell:
    def __init__(self, b : float, h : float) -> None:
        self.b = b
        self.h = h

class FitnessFunctionConfig:
    def __init__(self, alpha : float, beta : float, gamma : float) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

class ComparisonFunctionConfig:
    def __init__(self,
        fitness_function_config : FitnessFunctionConfig,
        p : float,
        sigma : float
    ) -> None:
        self.fitness_function_config = fitness_function_config
        self.p = p
        self.sigma = sigma

class SectorBoundaries:
    def __init__(self, boundaries : list[int]) -> None:
        """Initialize sector boundaries object.

        Each entry boundaries[i] has the meaning:
        "There is a boundary after boundaries[i]."

        For example, if the boundary is 3 | 4, the
        corresponding entry in boundaries would be 3.
        """
        self.boundaries = boundaries

# Helpers for simulation

def f(cell : Cell, fitness_function_config : FitnessFunctionConfig) -> float:
    """Fitness function."""
    term_1 = fitness_function_config.alpha * (cell.b - cell.h) ** 2
    term_2 = fitness_function_config.beta * (cell.b - cell.h) ** 4
    term_3 = fitness_function_config.gamma * (cell.b - cell.h)
    return term_1 + term_2 + term_3

def g(
    cell_left : Cell,
    cell_right : Cell,
    comparison_function_config : ComparisonFunctionConfig
) -> Cell:
    """Comparison function."""
    # Calculate left and right fitness
    fitness_left = f(
        cell=cell_left,
        fitness_function_config=comparison_function_config.fitness_function_config
    )

    fitness_right = f(
        cell=cell_right,
        fitness_function_config=comparison_function_config.fitness_function_config
    )

    # With probability p, bigger fitness reproduces
    if np.random.random_sample() <= comparison_function_config.p:
        # Bigger fitness reproduces
        if fitness_left > fitness_right:
            new_b = cell_left.b
            new_h = cell_left.h
        elif fitness_left == fitness_right:
            if np.random.random_sample() < 0.5:
                new_b = cell_left.b
                new_h = cell_left.h
            else:
                new_b = cell_right.b
                new_h = cell_right.h
        else:
            new_b = cell_right.b
            new_h = cell_right.h
    else:
        # Smaller fitness reproduces
        if fitness_left < fitness_right:
            new_b = cell_left.b
            new_h = cell_left.h
        elif fitness_left == fitness_right:
            if np.random.random_sample() < 0.5:
                new_b = cell_left.b
                new_h = cell_left.h
            else:
                new_b = cell_right.b
                new_h = cell_right.h
        else:
            new_b = cell_right.b
            new_h = cell_right.h
    
    # Add noise to new b and new h
    new_b += np.random.normal(0, comparison_function_config.sigma)
    new_h += np.random.normal(0, comparison_function_config.sigma)

    # Create new cell and return
    return Cell(b=new_b, h=new_h)

def propagate(
    curr_generation : list[Cell],
    is_even_generation : bool,
    comparison_function_config : ComparisonFunctionConfig
) -> list[Cell]:
    """Given a generation, get the next generation."""
    N = len(curr_generation)

    if is_even_generation:
        new_generation = []
        for i in range(N):
            new_generation.append(
                g(
                    cell_left=curr_generation[i],
                    cell_right=curr_generation[(i + 1) % N],
                    comparison_function_config=comparison_function_config
                )
            )
    else:
        new_generation = []
        for i in range(N):
            new_generation.append(
                g(
                    cell_left=curr_generation[(i - 1) % N],
                    cell_right=curr_generation[i],
                    comparison_function_config=comparison_function_config
                )
            )
    
    return new_generation

# Simulation functions

def simulate_until_last_generation(
    zeroth_generation : list[Cell],
    comparison_function_config : ComparisonFunctionConfig,
    last_generation_number : int
) -> list[list[Cell]]:
    """Simulate until the last generation is generated."""
    generations = []

    curr_generation = zeroth_generation
    is_even_generation = True
    for i in range(last_generation_number):
        # Append generation i
        #print(f"Appending generation {i}")
        #print(f"Generation {i} is even: {is_even_generation}")
        generations.append(curr_generation)

        # Produce generation i+1
        curr_generation = propagate(
            curr_generation=curr_generation,
            is_even_generation=is_even_generation,
            comparison_function_config=comparison_function_config
        )
        is_even_generation = not is_even_generation
    
    # Append generation T
    #print(f"Appending generation {last_generation_number}")
    #print(f"Generation {last_generation_number} is even: {is_even_generation}")
    generations.append(curr_generation)

    return generations

def convert_cells_to_heatmap(
    cell_generations : list[list[Cell]]
) -> list[list[float]]:
    """Convert Cells to their m value."""
    return [[cell.b - cell.h for cell in gen] for gen in cell_generations]

def reached_fixation(curr_generation : list[Cell]) -> bool:
    """Return true if b > h or b < h for all cells in generation."""
    b_dominant = all([cell.b > cell.h for cell in curr_generation])
    h_dominant = all([cell.b < cell.h for cell in curr_generation])
    return b_dominant or h_dominant

def simulate_until_fixation(
    zeroth_generation : list[Cell],
    comparison_function_config : ComparisonFunctionConfig,
) -> int:
    """Simulate until fixation."""
    curr_generation = zeroth_generation
    is_even_generation = True
    curr_generation_number = 0
    while not reached_fixation(curr_generation):
        curr_generation = propagate(
            curr_generation=curr_generation,
            is_even_generation=is_even_generation,
            comparison_function_config=comparison_function_config
        )
        is_even_generation = not is_even_generation
        curr_generation_number += 1
    
    return curr_generation_number

# Statistics

def compute_sector_boundaries_generation(curr_generation : list[Cell]) -> SectorBoundaries:
    """Compute sector boundaries for this generation."""
    N = len(curr_generation)
    boundaries = []
    for i in range(N):
        curr_cell_index = i
        next_cell_index = (i + 1) % N
        curr_cell = curr_generation[curr_cell_index]
        next_cell = curr_generation[next_cell_index]

        curr_cell_m = curr_cell.b - curr_cell.h
        next_cell_m = next_cell.b - next_cell.h

        if curr_cell_m * next_cell_m <= 0:
            boundaries.append(i)
    return SectorBoundaries(boundaries=boundaries)

def compute_sector_boundaries_trajectory(trajectory : list[list[Cell]]) -> list[SectorBoundaries]:
    """Compute sector boundaries for a trajectory."""
    return [compute_sector_boundaries_generation(curr_generation=generation) for generation in trajectory]

# (can also add more statistics like m and stuff)

# Special case sector boundary

def find_special_case_sector_boundary(
    curr_generation : list[Cell],
    is_even_generation : bool,
    next_generation : list[Cell],
    curr_sector_boundary : int,
) -> Optional[int]:
    """Get next sector boundary, given the current one."""
    curr_sector_boundaries = compute_sector_boundaries_generation(curr_generation=curr_generation)
    next_sector_boundaries = compute_sector_boundaries_generation(curr_generation=next_generation)

    # Sanity check
    assert curr_sector_boundary in curr_sector_boundaries.boundaries

    if is_even_generation:
        sector_boundary_candidates = [curr_sector_boundary - 1, curr_sector_boundary]
    else:
        sector_boundary_candidates = [curr_sector_boundary, curr_sector_boundary + 1]

    for sector_boundary_candidate in sector_boundary_candidates:
        if sector_boundary_candidate in next_sector_boundaries.boundaries:
            return sector_boundary_candidate

def trace_special_case_sector_boundaries(trajectory : list[list[Cell]], zeroth_sector_boundary : int) -> list[int]:
    """Trace special case sector boundary through a trajectory."""

    # Sanity check
    assert zeroth_sector_boundary in compute_sector_boundaries_generation(curr_generation=trajectory[0]).boundaries

    sector_boundary_trajectory = [zeroth_sector_boundary]
    T = len(trajectory)
    for i in range(T - 1):
        if len(compute_sector_boundaries_generation(curr_generation=trajectory[i]).boundaries) < 2:
            break
    #while len(compute_sector_boundaries_generation(curr_generation=trajectory[i]).boundaries) == 2 and i < T - 1:
        curr_generation = trajectory[i]
        is_even_generation = i % 2 == 0
        next_generation = trajectory[i + 1]
        curr_sector_boundary = sector_boundary_trajectory[-1]
        
        result = find_special_case_sector_boundary(
                curr_generation=curr_generation,
                is_even_generation=is_even_generation,
                next_generation=next_generation,
                curr_sector_boundary=curr_sector_boundary,
            )
        if result is None:
            break
        sector_boundary_trajectory.append(result)
            
    return sector_boundary_trajectory

def sector_boundary_fluctuations(boundary: list[int], zeroth_sector_boundary : int) -> float:
    """Calculate variance of sector boundary."""
    N = len(boundary)
    return sum((boundary[i] - zeroth_sector_boundary) ** 2 for i in range(N))/N

def sector_boundary_fluctuations_time_series(boundary : list[int], 
                                             zeroth_sector_boundary : int) -> list[float]:
    return [sector_boundary_fluctuations(boundary[:t], zeroth_sector_boundary) for t in range(1,len(boundary))]


# Many trials
def single_transverse_fluctuations(zeroth_generation : list[Cell],
                                   zeroth_sector_boundary : int,
                                   comparison_function_config : ComparisonFunctionConfig,
                                   number_of_generations : int) -> list[float]:
    res = simulate_until_last_generation(zeroth_generation=zeroth_generation,
                                comparison_function_config=comparison_function_config,
                                last_generation_number=number_of_generations)
    sector_boundary = trace_special_case_sector_boundaries(res, zeroth_sector_boundary)
    return sector_boundary_fluctuations_time_series(sector_boundary, zeroth_sector_boundary)

def many_transverse_fluctuations(number_of_trials : int,
                                     zeroth_generation : list[Cell],
                                     zeroth_sector_boundary : int,
                                     comparison_function_config : ComparisonFunctionConfig,
                                     number_of_generations : int) -> list[float]:
    all_results = []
    for i in range(number_of_trials):
        print(f'trial #{i}', end = '\r')
        result = single_transverse_fluctuations(zeroth_generation=zeroth_generation,
                                                zeroth_sector_boundary=zeroth_sector_boundary,
                                       comparison_function_config=comparison_function_config,
                                       number_of_generations=number_of_generations)
        all_results.append(result)
    return all_results

if __name__ == "__main__":
    pass
