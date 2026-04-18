import requests
import json
import time
import os
import speech_recognition as sr
from dotenv import load_dotenv
from obj_det import locate_and_segment

load_dotenv()

def get_voice_command():
    """
    Uses the system microphone to capture a voice command and translates it to text.
    Includes retries, confirmation, and text fallback.
    """
    recognizer = sr.Recognizer()

    # Tuned thresholds make speech capture less sensitive to background noise.
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8
    recognizer.non_speaking_duration = 0.4

    max_attempts = 3
    record_seconds = 6

    for attempt in range(1, max_attempts + 1):
        with sr.Microphone() as source:
            print("\n[Microphone] Adjusting for ambient noise... Please wait.")
            recognizer.adjust_for_ambient_noise(source, duration=0.8)
            print(f"\n[Microphone] Attempt {attempt}/{max_attempts}")
            print(f"[Microphone] Recording for {record_seconds} seconds... Speak now!")
            try:
                audio = recognizer.record(source, duration=record_seconds)
                print("[Microphone] Processing speech...")
                # Use Google's free Web Speech API (no API key required)
                text = recognizer.recognize_google(audio).strip()

                if not text:
                    print("[Microphone] Empty speech result. Please try again.")
                    continue

                print(f"[Voice Input] You said: '{text}'")
                confirm = input("Use this command? (y/n): ").strip().lower()
                if confirm == "y":
                    return text

                change_len = input("Change recording length? Enter seconds (or press Enter to keep 6): ").strip()
                if change_len.isdigit() and 2 <= int(change_len) <= 15:
                    record_seconds = int(change_len)

            except sr.UnknownValueError:
                print("[Microphone] Could not understand audio. Please speak clearly and closer to mic.")
            except sr.RequestError as e:
                print(f"[Microphone] Could not request speech results: {e}")
                break
            except Exception as e:
                print(f"[Microphone] Unexpected audio error: {e}")

        if attempt < max_attempts:
            retry = input("Try voice capture again? (y/n): ").strip().lower()
            if retry != "y":
                break

    print("[Microphone] Voice input unsuccessful.")
    return None


def get_text_command():
    """Fallback text input for task description."""
    task_description = input("Enter the task for the robot (e.g., 'pick the green tape only'): ")
    if not task_description.strip():
        print("No task entered. Exiting.")
        return None
    return task_description

def get_llm_plan(task_description):
    """
    Queries an LLM to break down a high-level task into a structured JSON plan.
    """
    prompt_text = f"""
    Convert the following user task into a structured JSON array of robotic actions.
    Allowed actions:
    - "grasp" (requires "target" string)
    - "move" (requires "target" string, destination/location)
    - "drop" (requires "location" string)
    - "wait" (optional "seconds" number, default 1)
    - "verify" (requires "target" string, optional "strict" boolean)
    - "repeat" (requires "count" integer and "steps" array of actions)

    Rules:
    - Decompose complex instructions into many small ordered steps.
    - Preserve object descriptors (color/material/size), e.g., "green tape".
    - If user mentions multiple objects or destinations, create separate sequences for each.
    - Output ONLY JSON, no markdown or explanations.
    
    User task: "{task_description}"
    
    Return ONLY valid JSON, without any conversational text or markdown blocks.
    Example output format: 
    [
      {{"action": "grasp", "target": "red apple"}}, 
      {{"action": "move", "target": "basket"}}, 
      {{"action": "drop", "location": "basket"}}
    ]
    """
    
    print(f"\n[LLM Request] Task: {task_description}")
    print("[LLM Processing] Thinking...")
    
    response = requests.post(
      url="https://openrouter.ai/api/v1/chat/completions",
      headers={
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json"
      },
      data=json.dumps({
        "model": "arcee-ai/trinity-large-preview:free",
        "messages": [
          {
            "role": "user",
            "content": prompt_text
          }
        ]
      })
    )

    try:
        data = response.json()
        if response.status_code != 200:
            print(f"Error from API (HTTP {response.status_code}): {data}")
            return []

        if "choices" not in data or not data["choices"]:
            print(f"Error from API: {data}")
            return []

        message = data['choices'][0]['message']['content'].strip()
        
        # Clean up the message if returned with markdown code blocks
        if "```json" in message:
            message = message.split("```json")[1].split("```")[0].strip()
        elif "```" in message:
            message = message.split("```")[1].strip()
            
        plan = normalize_and_expand_plan(json.loads(message))
        print("\n[LLM Response] Plan generated successfully:")
        print(json.dumps(plan, indent=2))
        return plan
        
    except json.JSONDecodeError as e:
        print(f"Error parsing LLM JSON: {e}")
        print(f"Raw Output: {message}")
        return []
    except Exception as e:
        print(f"Error connecting to LLM: {e}")
        return []


def normalize_and_expand_plan(plan_raw):
    """
    Normalizes LLM plan output and expands repeat steps for deterministic execution.
    """
    if isinstance(plan_raw, dict):
        plan_raw = plan_raw.get("plan", [])

    if not isinstance(plan_raw, list):
        return []

    action_alias = {
        "pick": "grasp",
        "grab": "grasp",
        "pickup": "grasp",
        "pick_up": "grasp",
        "go": "move",
        "navigate": "move",
        "place": "drop",
        "release": "drop",
        "put": "drop",
        "putdown": "drop",
        "put_down": "drop",
        "pause": "wait",
        "sleep": "wait",
        "check": "verify",
        "confirm": "verify",
        "inspect": "verify",
        "look_for": "verify",
    }

    normalized = []
    for step in plan_raw:
        if not isinstance(step, dict):
            continue

        raw_action = str(step.get("action", "")).strip().lower()
        action = action_alias.get(raw_action, raw_action)

        if action == "repeat":
            count = step.get("count", 1)
            try:
                count = max(1, int(count))
            except Exception:
                count = 1

            nested_steps = normalize_and_expand_plan(step.get("steps", []))
            for _ in range(count):
                normalized.extend(nested_steps)
            continue

        if action == "grasp":
            target = step.get("target")
            if isinstance(target, str) and target.strip():
                normalized.append({"action": "grasp", "target": target.strip()})

        elif action == "move":
            target = step.get("target") or step.get("location")
            if isinstance(target, str) and target.strip():
                normalized.append({"action": "move", "target": target.strip()})

        elif action == "drop":
            location = step.get("location") or step.get("target")
            if isinstance(location, str) and location.strip():
                normalized.append({"action": "drop", "location": location.strip()})

        elif action == "wait":
            seconds = step.get("seconds", 1)
            try:
                seconds = max(0.2, min(30.0, float(seconds)))
            except Exception:
                seconds = 1.0
            normalized.append({"action": "wait", "seconds": seconds})

        elif action == "verify":
            target = step.get("target")
            if isinstance(target, str) and target.strip():
                normalized.append(
                    {
                        "action": "verify",
                        "target": target.strip(),
                        "strict": bool(step.get("strict", False)),
                    }
                )

    return normalized

def execute_plan(plan):
    """
    Loops over the structured actions and executes them.
    """
    held_object = None

    for step_index, step in enumerate(plan):
        action = step.get("action")
        target = step.get("target") or step.get("location")
        
        print(f"\n--- {step_index+1}. Executing Step: {action.upper()} '{target}' ---")
        
        if action == "grasp":
            # Call vision module to locate the object dynamically at this exact moment
            print(f"Robot locating '{target}' using camera...")
            coords = locate_and_segment(target)
            
            if coords:
                print(f"[Robot Hardware] Moving arm to coordinates: X:{coords[0]}, Y:{coords[1]}")
                time.sleep(1)
                print(f"[Robot Hardware] Closing gripper. '{target}' grasped successfully.")
                held_object = target
            else:
                print(f"[Robot Hardware Error] Could not find '{target}'. Aborting sequence.")
                break
                
        elif action == "move":
            print(f"[Robot Hardware] Moving arm towards '{target}'...")
            time.sleep(1)
            print(f"[Robot Hardware] Position reached.")
            
        elif action == "drop":
            print(f"[Robot Hardware] Opening gripper. Dropping payload into '{target}'.")
            time.sleep(0.5)
            held_object = None

        elif action == "wait":
            seconds = step.get("seconds", 1)
            print(f"[Robot Hardware] Waiting for {seconds:.1f} seconds...")
            time.sleep(seconds)

        elif action == "verify":
            strict = bool(step.get("strict", False))
            print(f"[Robot Vision] Verifying visibility of '{target}'...")
            coords = locate_and_segment(target)
            if coords:
                print(f"[Robot Vision] Verified '{target}' at X:{coords[0]}, Y:{coords[1]}.")
            else:
                print(f"[Robot Vision Warning] Could not verify '{target}'.")
                if strict:
                    print("[Robot Execution] Strict verify failed. Aborting sequence.")
                    break
            
        else:
            print(f"[Error] Unknown action type: {action}")
            
        time.sleep(1) # Pause between steps

    if held_object:
        print(f"[Robot Notice] Sequence ended while holding '{held_object}'.")

def main():
    print("=== Robot Task Orchestrator Initialized ===")
    
    # 1. Prompt user whether they want to use microphone or typing
    use_voice_input = input("Would you like to use voice commands? (y/n): ").strip().lower()
    
    if use_voice_input == 'y':
        task_description = get_voice_command()
        if not task_description:
            fallback = input("Switch to typing the command instead? (y/n): ").strip().lower()
            if fallback == "y":
                task_description = get_text_command()
                if not task_description:
                    return
            else:
                return
    else:
        task_description = get_text_command()
        if not task_description:
            return
    
    # 2. Chain-of-Thought / LLM Planning
    plan = get_llm_plan(task_description)
    
    # 3. Execution Loop
    if plan:
        execute_plan(plan)
    
    print("\n=== Task Complete ===")

if __name__ == "__main__":
    main()
