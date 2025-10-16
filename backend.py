import time
import json
import numpy as np
from enum import Enum
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import deque
import random
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EmotionEngine")

class EmotionalState(Enum):
    CALM = "calm"
    ANXIOUS = "anxious"
    FRUSTRATED = "frustrated"
    BORED = "bored"
    TENSE = "tense"
    SURPRISED = "surprised"
    ENGAGED = "engaged"

class NarrativeTone(Enum):
    NEUTRAL = "neutral"
    HORROR = "horror"
    COMFORT = "comfort"
    MYSTERY = "mystery"
    URGENCY = "urgency"
    RELAXED = "relaxed"

@dataclass
class PlayerInput:
    timestamp: float
    mouse_movement_speed: float  # pixels per second
    mouse_click_frequency: float  # clicks per minute
    reaction_time: float  # seconds
    input_regularity: float  # std dev of time between inputs
    camera_detected_emotion: Optional[str] = None
    in_game_behavior: Dict[str, Any] = None

@dataclass
class EmotionalMetrics:
    state: EmotionalState
    confidence: float
    arousal: float  # 0-1 scale (low to high)
    valence: float  # 0-1 scale (negative to positive)

@dataclass
class NarrativeEvent:
    event_id: str
    description: str
    target_emotion: EmotionalState
    tone: NarrativeTone
    music_track: str
    background_art: str
    lighting: str
    dialogue: str
    intensity: float  # 0-1 scale

class PlayerStateInference:
    def __init__(self):
        self.input_history = deque(maxlen=100)  # Store last 100 inputs
        self.emotion_history = deque(maxlen=50)  # Store emotion history
        self.current_state = EmotionalState.CALM
        self.state_confidence = 0.0
        
        # Emotion model thresholds (would normally be ML model)
        self.thresholds = {
            "boredom": {
                "mouse_speed_max": 50.0,
                "click_frequency_max": 2.0,
                "reaction_time_min": 2.0
            },
            "anxiety": {
                "mouse_speed_min": 150.0,
                "click_frequency_min": 10.0,
                "reaction_time_max": 0.5
            },
            "frustration": {
                "mouse_speed_irregular": 0.8,  # High variance
                "click_frequency_high": 15.0,
                "rapid_input_changes": 5  # Number of direction changes
            }
        }
        
    def process_input(self, player_input: PlayerInput) -> EmotionalMetrics:
        """Analyze player input to infer emotional state"""
        self.input_history.append(player_input)
        
        if len(self.input_history) < 5:  # Need minimum data points
            return EmotionalMetrics(
                state=EmotionalState.CALM,
                confidence=0.3,
                arousal=0.3,
                valence=0.5
            )
        
        # Calculate metrics from input history
        metrics = self._calculate_emotional_metrics()
        
        # Update current state
        self.current_state = metrics.state
        self.state_confidence = metrics.confidence
        
        self.emotion_history.append(metrics)
        logger.info(f"Inferred emotional state: {metrics.state} (confidence: {metrics.confidence:.2f})")
        
        return metrics
    
    def _calculate_emotional_metrics(self) -> EmotionalMetrics:
        """Calculate emotional metrics from input patterns"""
        recent_inputs = list(self.input_history)[-10:]  # Last 10 inputs
        
        # Calculate basic metrics
        mouse_speeds = [inp.mouse_movement_speed for inp in recent_inputs]
        click_freqs = [inp.mouse_click_frequency for inp in recent_inputs]
        reaction_times = [inp.reaction_time for inp in recent_inputs]
        
        avg_mouse_speed = np.mean(mouse_speeds)
        avg_click_freq = np.mean(click_freqs)
        avg_reaction_time = np.mean(reaction_times)
        mouse_speed_std = np.std(mouse_speeds)
        
        # Emotional state inference logic
        state_scores = {
            EmotionalState.BORED: self._calculate_boredom_score(avg_mouse_speed, avg_click_freq, avg_reaction_time),
            EmotionalState.ANXIOUS: self._calculate_anxiety_score(avg_mouse_speed, avg_click_freq, avg_reaction_time),
            EmotionalState.FRUSTRATED: self._calculate_frustration_score(mouse_speed_std, avg_click_freq),
            EmotionalState.CALM: self._calculate_calm_score(avg_mouse_speed, avg_click_freq, avg_reaction_time),
        }
        
        # Determine most likely state
        likely_state = max(state_scores, key=state_scores.get)
        confidence = state_scores[likely_state]
        
        # Calculate arousal and valence
        arousal = self._calculate_arousal(avg_mouse_speed, avg_click_freq)
        valence = self._calculate_valence(avg_reaction_time, mouse_speed_std)
        
        return EmotionalMetrics(
            state=likely_state,
            confidence=confidence,
            arousal=arousal,
            valence=valence
        )
    
    def _calculate_boredom_score(self, mouse_speed, click_freq, reaction_time):
        score = 0.0
        if mouse_speed < self.thresholds["boredom"]["mouse_speed_max"]:
            score += 0.4
        if click_freq < self.thresholds["boredom"]["click_frequency_max"]:
            score += 0.3
        if reaction_time > self.thresholds["boredom"]["reaction_time_min"]:
            score += 0.3
        return score
    
    def _calculate_anxiety_score(self, mouse_speed, click_freq, reaction_time):
        score = 0.0
        if mouse_speed > self.thresholds["anxiety"]["mouse_speed_min"]:
            score += 0.4
        if click_freq > self.thresholds["anxiety"]["click_frequency_min"]:
            score += 0.3
        if reaction_time < self.thresholds["anxiety"]["reaction_time_max"]:
            score += 0.3
        return score
    
    def _calculate_frustration_score(self, mouse_speed_std, click_freq):
        score = 0.0
        if mouse_speed_std > self.thresholds["frustration"]["mouse_speed_irregular"]:
            score += 0.5
        if click_freq > self.thresholds["frustration"]["click_frequency_high"]:
            score += 0.5
        return score
    
    def _calculate_calm_score(self, mouse_speed, click_freq, reaction_time):
        # Calm is medium mouse speed, medium click frequency, medium reaction time
        ideal_mouse_range = (80.0, 120.0)
        ideal_click_range = (3.0, 8.0)
        ideal_reaction_range = (0.8, 1.5)
        
        score = 0.0
        if ideal_mouse_range[0] <= mouse_speed <= ideal_mouse_range[1]:
            score += 0.4
        if ideal_click_range[0] <= click_freq <= ideal_click_range[1]:
            score += 0.3
        if ideal_reaction_range[0] <= reaction_time <= ideal_reaction_range[1]:
            score += 0.3
        return score
    
    def _calculate_arousal(self, mouse_speed, click_freq):
        # Higher mouse speed and click frequency = higher arousal
        normalized_speed = min(mouse_speed / 200.0, 1.0)  # Cap at 200 px/s
        normalized_clicks = min(click_freq / 20.0, 1.0)   # Cap at 20 clicks/min
        return (normalized_speed + normalized_clicks) / 2.0
    
    def _calculate_valence(self, reaction_time, mouse_std):
        # Lower reaction time and more consistent movement = more positive valence
        rt_valence = 1.0 - min(reaction_time / 3.0, 1.0)  # Longer RT = more negative
        consistency_valence = 1.0 - min(mouse_std / 2.0, 1.0)  # More erratic = more negative
        return (rt_valence + consistency_valence) / 2.0

class DynamicNarrativeDirector:
    def __init__(self, target_emotional_arc: List[EmotionalState]):
        self.target_arc = target_emotional_arc
        self.current_arc_index = 0
        self.target_emotion = target_emotional_arc[0]
        self.intervention_history = []
        self.narrative_events = self._initialize_narrative_events()
        
    def _initialize_narrative_events(self) -> Dict[str, NarrativeEvent]:
        """Initialize a library of narrative events for different emotional tones"""
        return {
            "horror_intense": NarrativeEvent(
                event_id="horror_intense",
                description="Intense horror scene with jump scare",
                target_emotion=EmotionalState.ANXIOUS,
                tone=NarrativeTone.HORROR,
                music_track="tense_horror_theme",
                background_art="dark_corridor",
                lighting="flickering_lights",
                dialogue="You hear whispers getting closer...",
                intensity=0.9
            ),
            "horror_mild": NarrativeEvent(
                event_id="horror_mild",
                description="Mild horror atmosphere",
                target_emotion=EmotionalState.TENSE,
                tone=NarrativeTone.HORROR,
                music_track="eerie_ambient",
                background_art="foggy_forest",
                lighting="dim_blue",
                dialogue="The air grows cold around you.",
                intensity=0.6
            ),
            "comfort_calm": NarrativeEvent(
                event_id="comfort_calm",
                description="Calming comforting scene",
                target_emotion=EmotionalState.CALM,
                tone=NarrativeTone.COMFORT,
                music_track="peaceful_melody",
                background_art="safe_haven",
                lighting="warm_glow",
                dialogue="You feel a sense of peace and safety here.",
                intensity=0.3
            ),
            "mystery_intrigue": NarrativeEvent(
                event_id="mystery_intrigue",
                description="Mysterious discovery",
                target_emotion=EmotionalState.ENGAGED,
                tone=NarrativeTone.MYSTERY,
                music_track="mysterious_theme",
                background_art="ancient_ruins",
                lighting="moonlight",
                dialogue="What secrets does this place hold?",
                intensity=0.7
            ),
            "urgency_action": NarrativeEvent(
                event_id="urgency_action",
                description="Urgent action sequence",
                target_emotion=EmotionalState.FRUSTRATED,
                tone=NarrativeTone.URGENCY,
                music_track="fast_paced_action",
                background_art="collapsing_bridge",
                lighting="bright_red",
                dialogue="Quick! There's not much time!",
                intensity=0.8
            )
        }
    
    def get_narrative_intervention(self, current_emotion: EmotionalMetrics, 
                                 time_in_current_state: float) -> Optional[NarrativeEvent]:
        """Determine if and what narrative intervention is needed"""
        
        # Check if we should progress the emotional arc
        if (current_emotion.state == self.target_emotion and 
            current_emotion.confidence > 0.6 and
            time_in_current_state > 30.0):  # Stay in target emotion for 30 seconds
            self._progress_emotional_arc()
        
        # Calculate emotional distance from target
        emotional_distance = self._calculate_emotional_distance(current_emotion, self.target_emotion)
        
        # Only intervene if significantly off-target and confident about current state
        if emotional_distance > 0.5 and current_emotion.confidence > 0.5:
            intervention = self._select_appropriate_intervention(current_emotion, emotional_distance)
            if intervention:
                self.intervention_history.append({
                    "timestamp": time.time(),
                    "current_emotion": current_emotion.state.value,
                    "target_emotion": self.target_emotion.value,
                    "intervention": intervention.event_id,
                    "emotional_distance": emotional_distance
                })
                logger.info(f"Intervention triggered: {intervention.event_id}")
                return intervention
        
        return None
    
    def _calculate_emotional_distance(self, current: EmotionalMetrics, target: EmotionalState) -> float:
        """Calculate how far current emotion is from target emotion"""
        # Simple mapping of emotional states to arousal-valence coordinates
        emotion_coordinates = {
            EmotionalState.CALM: (0.3, 0.7),
            EmotionalState.ANXIOUS: (0.8, 0.3),
            EmotionalState.FRUSTRATED: (0.7, 0.2),
            EmotionalState.BORED: (0.2, 0.4),
            EmotionalState.TENSE: (0.6, 0.4),
            EmotionalState.SURPRISED: (0.9, 0.6),
            EmotionalState.ENGAGED: (0.5, 0.8)
        }
        
        current_coord = (current.arousal, current.valence)
        target_coord = emotion_coordinates[target]
        
        # Euclidean distance in arousal-valence space
        distance = np.sqrt(
            (current_coord[0] - target_coord[0])**2 + 
            (current_coord[1] - target_coord[1])**2
        )
        
        return distance
    
    def _select_appropriate_intervention(self, current_emotion: EmotionalMetrics, 
                                       distance: float) -> Optional[NarrativeEvent]:
        """Select the most appropriate narrative intervention"""
        
        # Filter events by target emotion
        candidate_events = [event for event in self.narrative_events.values() 
                          if event.target_emotion == self.target_emotion]
        
        if not candidate_events:
            return None
        
        # Select event based on intensity needed
        if distance > 0.8:  # Very far from target - use strong intervention
            intense_events = [e for e in candidate_events if e.intensity > 0.7]
            return random.choice(intense_events) if intense_events else candidate_events[0]
        elif distance > 0.6:  # Moderately far - medium intervention
            medium_events = [e for e in candidate_events if 0.4 <= e.intensity <= 0.7]
            return random.choice(medium_events) if medium_events else candidate_events[0]
        else:  # Somewhat far - subtle intervention
            subtle_events = [e for e in candidate_events if e.intensity < 0.4]
            return random.choice(subtle_events) if subtle_events else candidate_events[0]
    
    def _progress_emotional_arc(self):
        """Move to the next emotion in the target emotional arc"""
        if self.current_arc_index < len(self.target_arc) - 1:
            self.current_arc_index += 1
            self.target_emotion = self.target_arc[self.current_arc_index]
            logger.info(f"Progressing emotional arc to: {self.target_emotion}")

class EmotionalJourneyTracker:
    def __init__(self):
        self.session_data = {
            "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "start_time": time.time(),
            "emotional_journey": [],
            "interventions": [],
            "final_visualization": None
        }
    
    def record_emotional_state(self, emotional_metrics: EmotionalMetrics):
        """Record a snapshot of the player's emotional state"""
        record = {
            "timestamp": time.time(),
            "emotion": emotional_metrics.state.value,
            "confidence": emotional_metrics.confidence,
            "arousal": emotional_metrics.arousal,
            "valence": emotional_metrics.valence
        }
        self.session_data["emotional_journey"].append(record)
    
    def record_intervention(self, intervention: NarrativeEvent, emotional_state_before: EmotionalMetrics):
        """Record an intervention and the emotional state before it"""
        record = {
            "timestamp": time.time(),
            "intervention_id": intervention.event_id,
            "emotion_before": emotional_state_before.state.value,
            "target_emotion": intervention.target_emotion.value
        }
        self.session_data["interventions"].append(record)
    
    def generate_visualization(self):
        """Generate a visualization of the emotional journey"""
        # This would typically create charts/graphs
        # For this prototype, we'll create a text-based visualization
        
        emotions = [entry["emotion"] for entry in self.session_data["emotional_journey"]]
        interventions = self.session_data["interventions"]
        
        visualization = {
            "session_duration": time.time() - self.session_data["start_time"],
            "emotional_transitions": len(set(emotions)),
            "most_prevalent_emotion": max(set(emotions), key=emotions.count),
            "intervention_count": len(interventions),
            "intervention_success_rate": self._calculate_intervention_success_rate(),
            "emotional_arc": emotions
        }
        
        self.session_data["final_visualization"] = visualization
        return visualization
    
    def _calculate_intervention_success_rate(self) -> float:
        """Calculate how often interventions led to desired emotional change"""
        if not self.session_data["interventions"]:
            return 0.0
        
        successful_interventions = 0
        interventions = self.session_data["interventions"]
        emotional_journey = self.session_data["emotional_journey"]
        
        for i, intervention in enumerate(interventions):
            # Find emotional state before and after intervention
            intervention_time = intervention["timestamp"]
            target_emotion = intervention["target_emotion"]
            
            # Look for emotional state 30 seconds after intervention
            post_intervention_states = [
                entry for entry in emotional_journey 
                if entry["timestamp"] > intervention_time and 
                entry["timestamp"] < intervention_time + 30
            ]
            
            if post_intervention_states:
                # Check if any post-intervention state matches target
                for state in post_intervention_states:
                    if state["emotion"] == target_emotion and state["confidence"] > 0.5:
                        successful_interventions += 1
                        break
        
        return successful_interventions / len(interventions) if interventions else 0.0
    
    def save_session_data(self, filename: str = None):
        """Save session data to JSON file"""
        if filename is None:
            filename = f"emotional_journey_{self.session_data['session_id']}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.session_data, f, indent=2, default=str)
        
        logger.info(f"Session data saved to {filename}")

class EmotionallyAdaptiveGameEngine:
    def __init__(self, target_emotional_arc: List[EmotionalState]):
        self.player_state_inference = PlayerStateInference()
        self.narrative_director = DynamicNarrativeDirector(target_emotional_arc)
        self.journey_tracker = EmotionalJourneyTracker()
        self.last_emotion_change_time = time.time()
        self.current_emotion_duration = 0.0
        
        logger.info("Emotionally Adaptive Game Engine initialized")
    
    def process_game_tick(self, player_input: PlayerInput) -> Optional[NarrativeEvent]:
        """Process one tick of the game engine"""
        
        # Infer current emotional state
        emotional_state = self.player_state_inference.process_input(player_input)
        
        # Track emotional journey
        self.journey_tracker.record_emotional_state(emotional_state)
        
        # Update emotion duration tracking
        self._update_emotion_duration(emotional_state.state)
        
        # Check if narrative intervention is needed
        intervention = self.narrative_director.get_narrative_intervention(
            emotional_state, self.current_emotion_duration
        )
        
        if intervention:
            self.journey_tracker.record_intervention(intervention, emotional_state)
        
        return intervention
    
    def _update_emotion_duration(self, new_emotion: EmotionalState):
        """Track how long the player has been in the current emotional state"""
        current_time = time.time()
        
        if (new_emotion != self.player_state_inference.current_state or 
            current_time - self.last_emotion_change_time > 300):  # Reset after 5 minutes
            self.last_emotion_change_time = current_time
            self.current_emotion_duration = 0.0
        else:
            self.current_emotion_duration = current_time - self.last_emotion_change_time
    
    def end_game_session(self):
        """End the game session and generate final analytics"""
        visualization = self.journey_tracker.generate_visualization()
        self.journey_tracker.save_session_data()
        
        logger.info("Game session ended. Emotional journey visualization:")
        logger.info(json.dumps(visualization, indent=2))
        
        return visualization

# Example usage and simulation
def simulate_game_session():
    """Simulate a game session to demonstrate the engine"""
    
    # Define target emotional arc for a horror game
    target_arc = [
        EmotionalState.CALM,      # Beginning - safe area
        EmotionalState.TENSE,     # Building tension
        EmotionalState.ANXIOUS,   # Horror peak
        EmotionalState.SURPRISED, # Plot twist
        EmotionalState.ENGAGED    # Resolution
    ]
    
    # Initialize engine
    game_engine = EmotionallyAdaptiveGameEngine(target_arc)
    
    # Simulate player inputs over time with changing emotional patterns
    simulation_duration = 300  # 5 minutes
    tick_interval = 5  # Process every 5 seconds
    
    print("Simulating game session...")
    
    for tick in range(0, simulation_duration, tick_interval):
        # Simulate different emotional input patterns over time
        if tick < 60:  # First minute: calm
            input_pattern = PlayerInput(
                timestamp=time.time(),
                mouse_movement_speed=90.0,
                mouse_click_frequency=5.0,
                reaction_time=1.2,
                input_regularity=0.3
            )
        elif tick < 120:  # Second minute: getting bored
            input_pattern = PlayerInput(
                timestamp=time.time(),
                mouse_movement_speed=40.0,
                mouse_click_frequency=1.5,
                reaction_time=2.5,
                input_regularity=0.8
            )
        elif tick < 180:  # Third minute: anxious
            input_pattern = PlayerInput(
                timestamp=time.time(),
                mouse_movement_speed=180.0,
                mouse_click_frequency=12.0,
                reaction_time=0.4,
                input_regularity=0.6
            )
        elif tick < 240:  # Fourth minute: frustrated
            input_pattern = PlayerInput(
                timestamp=time.time(),
                mouse_movement_speed=160.0,
                mouse_click_frequency=18.0,
                reaction_time=0.3,
                input_regularity=1.2  # High irregularity
            )
            
        else:  # Final minute: engaged
            input_pattern = PlayerInput(
                timestamp=time.time(),
                mouse_movement_speed=110.0,
                mouse_click_frequency=8.0,
                reaction_time=0.9,
                input_regularity=0.4
            )
        
        # Process game tick
        intervention = game_engine.process_game_tick(input_pattern)
        
        if intervention:
            print(f"Tick {tick}s: Intervention triggered - {intervention.event_id}")
        
        time.sleep(0.1)  # Small delay for realism
    
    # End session and get results
    results = game_engine.end_game_session()
    print("\nSimulation complete!")
    print(f"Session results: {json.dumps(results, indent=2)}")

if __name__ == "__main__":
    # Run simulation
    simulate_game_session()