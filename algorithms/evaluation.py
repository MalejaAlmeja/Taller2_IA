from __future__ import annotations
from algorithms.utils import bfs_distance, dijkstra
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from world.game_state import GameState


def evaluation_function(state: GameState) -> float:
    """
    Evaluation function for non-terminal states of the drone vs. hunters game.

    A good evaluation function can consider multiple factors, such as:
      (a) BFS distance from drone to nearest delivery point (closer is better).
          Uses actual path distance so walls and terrain are respected.
      (b) BFS distance from each hunter to the drone, traversing only normal
          terrain ('.' / ' ').  Hunters blocked by mountains, fog, or storms
          are treated as unreachable (distance = inf) and pose no threat.
      (c) BFS distance to a "safe" position (i.e., a position that is not in the path of any hunter).
      (d) Number of pending deliveries (fewer is better).
      (e) Current score (higher is better).
      (f) Delivery urgency: reward the drone for being close to a delivery it can
          reach strictly before any hunter, so it commits to nearby pickups
          rather than oscillating in place out of excessive hunter fear.
      (g) Adding a revisit penalty can help prevent the drone from getting stuck in cycles.

    Returns a value in [-1000, +1000].

    Tips:
    - Use state.get_drone_position() to get the drone's current (x, y) position.
    - Use state.get_hunter_positions() to get the list of hunter (x, y) positions.
    - Use state.get_pending_deliveries() to get the set of pending delivery (x, y) positions.
    - Use state.get_score() to get the current game score.
    - Use state.get_layout() to get the current layout.
    - Use state.is_win() and state.is_lose() to check terminal states.
    - Use bfs_distance(layout, start, goal, hunter_restricted) from algorithms.utils
      for cached BFS distances. hunter_restricted=True for hunter-only terrain.
    - Use dijkstra(layout, start, goal) from algorithms.utils for cached
      terrain-weighted shortest paths, returning (cost, path).
    - Consider edge cases: no pending deliveries, no hunters nearby.
    - A good evaluation function balances delivery progress with hunter avoidance.
    """
    
    score = 0
    if state.is_win():
        return 1000
    elif state.is_lose():
        return -1000

    drone_pos = state.get_drone_position()
    layout = state.get_layout() 
    
    delivery_distances = []
    hunter_distances = []
    safe_positions = set()
    safe_distances = []

    

    # 1. BFS distance from drone to nearest delivery point
    pending_deliveries = state.get_pending_deliveries()
    
    if pending_deliveries:
        for delivery in pending_deliveries:
            delivery_distances.append(bfs_distance(layout, drone_pos, delivery, hunter_restricted=False))
        min_delivery_distance = min(delivery_distances)
        score += 10 / (1 + min_delivery_distance)  # Closer deliveries increase score

    # 2. BFS distance from each hunter to the drone (considering only normal terrain
    hunters = state.get_hunter_positions()
    
    if hunters:
        for hunter in hunters:
            hunter_distances.append(bfs_distance(layout, drone_pos, hunter, hunter_restricted=True))
        min_hunter_distance = min(hunter_distances)
        score -= 10 / (1 + min_hunter_distance)  # Further hunters decrease score

    # 3. BFS distance to a "safe" position (not in the path of any hunter)
    for hunter in hunters:
        _, path = dijkstra(layout, hunter, drone_pos)
        safe_positions.update(path)

    for x in range(layout.width):
        for y in range(layout.height):
            pos = (x, y)
            if pos not in safe_positions and layout.get_terrain(x,y) != '%':  # Not impossible (%)
                safe_dist = bfs_distance(layout, drone_pos, pos, hunter_restricted=False)
                if safe_dist != float('inf'):
                    safe_distances.append(safe_dist)
    if safe_distances:
        min_safe_distance = min(safe_distances)
        score += 5 / (1 + min_safe_distance)  # Closer safe positions increase score
    
    # 4. Number of pending deliveries
    score -= len(pending_deliveries)

    # 5. Current score
    score += state.get_score()
    
    # 6. Delivery urgency (reward being close to a delivery that can be reached before any hunter)
    for delivery in pending_deliveries:
        drone_to_delivery = bfs_distance(layout, drone_pos, delivery, hunter_restricted=False)
        hunter_to_delivery = min(bfs_distance(layout, hunter, delivery, hunter_restricted=True) for hunter in hunters)
        if drone_to_delivery < hunter_to_delivery:
            score += 20 / (1 + drone_to_delivery)  # Closer urgent deliveries increase score
    
    # 7. Adding a revisit penalty can help prevent the drone from getting stuck in cycles.
    visit_count = state.get_visited_count(drone_pos)
    if visit_count > 1:
        score -= 5 * (visit_count - 1)  # each extra visit costs 5 points

    return score
