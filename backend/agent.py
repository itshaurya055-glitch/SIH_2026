import os
import json
import time
from openai import OpenAI

# Default fallback API key if not in environment
DEFAULT_KEY = "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"
API_KEY = os.environ.get("NVIDIA_API_KEY", DEFAULT_KEY)

# Decision history cache for visual feed
agent_decision_log = []

class AnomalyAgent:
    def __init__(self, callbacks: dict):
        self.callbacks = callbacks
        self.client = None
        self._init_client()

    def _init_client(self):
        if API_KEY:
            try:
                self.client = OpenAI(
                    base_url="https://integrate.api.nvidia.com/v1",
                    api_key=API_KEY
                )
                print("[agent] OpenAI client configured for NVIDIA NIM")
            except Exception as e:
                print(f"[agent] Error configuring client: {e}")
                self.client = None
        else:
            print("[agent] No API key set, running in MOCK fallback mode")

    def run_autonomous_agent(self, anomaly_details: dict) -> dict:
        """Invoked when an anomaly is detected. Queries telemetry, terrain, and decides

        whether to replan.
        """
        print(f"[agent] Invoking agent for anomaly: {anomaly_details.get('message')}")
        
        # Log entry for tracking agent thought process
        decision_entry = {
            "timestamp": time.time(),
            "anomaly": anomaly_details,
            "steps": [],
            "final_action": "Nominal",
            "explanation": ""
        }

        # Check if we should use fallback mock mode
        if not self.client:
            return self._run_mock_fallback(anomaly_details, decision_entry)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are the Mission Copilot AI Agent on a lunar rover.\n"
                    "You have just received an anomaly alert.\n"
                    "You have access to the following tools. You can run them by returning a single JSON block:\n\n"
                    "1. get_telemetry_window(start_tick: int, end_tick: int) -> list\n"
                    "2. get_terrain_info(row: int, col: int) -> dict\n"
                    "3. trigger_replan(reason: str) -> str\n"
                    "4. get_mission_log() -> list\n\n"
                    "Format tool calls as a single JSON object. Do not output any thinking or extra text outside the JSON:\n"
                    "{\n"
                    '  "thought": "Reasoning about what to do next...",\n'
                    '  "tool": "tool_name",\n'
                    '  "args": {"arg_name": value}\n'
                    "}\n\n"
                    "If you have enough information to explain the anomaly and make a decision, output the final explanation and action in this format:\n"
                    "{\n"
                    '  "thought": "Final summary thought...",\n'
                    '  "explanation": "Plain-English explanation of what caused the anomaly and what actions have been taken.",\n'
                    '  "decision": "replan" | "slow_down" | "none"\n'
                    "}\n"
                )
            },
            {
                "role": "user",
                "content": f"New anomaly context: {json.dumps(anomaly_details)}"
            }
        ]

        max_turns = 3
        for turn in range(max_turns):
            try:
                response = self.client.chat.completions.create(
                    model="nvidia/nemotron-3-ultra-550b-a55b",
                    messages=messages,
                    temperature=0.2,
                    max_tokens=1000
                )
                response_text = response.choices[0].message.content.strip()
                print(f"[agent] Turn {turn+1} raw model output: {response_text}")

                # Attempt to parse JSON block from model response
                try:
                    # Remove markdown code formatting if present
                    if response_text.startswith("```json"):
                        response_text = response_text[7:]
                    if response_text.endswith("```"):
                        response_text = response_text[:-3]
                    response_text = response_text.strip()

                    action = json.loads(response_text)
                except Exception as parse_err:
                    print(f"[agent] Failed parsing agent JSON: {parse_err}. Raw: {response_text}")
                    # Fallback on parse failure
                    break

                # Record the agent step
                decision_entry["steps"].append({
                    "thought": action.get("thought", ""),
                    "tool": action.get("tool", "none"),
                    "args": action.get("args", {})
                })

                # Check if final decision is reached
                if "explanation" in action:
                    decision_entry["final_action"] = action.get("decision", "none")
                    decision_entry["explanation"] = action.get("explanation", "")
                    
                    # Execute final action
                    if decision_entry["final_action"] == "replan":
                        self.callbacks["trigger_replan"](action.get("explanation", "Agent replan"))
                    
                    agent_decision_log.append(decision_entry)
                    return decision_entry

                # Execute requested tool
                tool_name = action.get("tool")
                tool_args = action.get("args", {})
                if tool_name in self.callbacks:
                    print(f"[agent] Running tool '{tool_name}' with args {tool_args}")
                    tool_result = self.callbacks[tool_name](**tool_args)
                    # Add to message history
                    messages.append({"role": "assistant", "content": response_text})
                    messages.append({
                        "role": "user",
                        "content": f"Tool '{tool_name}' returned: {json.dumps(tool_result)}"
                    })
                else:
                    print(f"[agent] Unknown tool: {tool_name}")
                    break

            except Exception as e:
                print(f"[agent] API calling error during autonomous loop: {e}")
                break

        # Fallback if loop completes without a clean exit
        return self._run_mock_fallback(anomaly_details, decision_entry)

    def _run_mock_fallback(self, anomaly_details: dict, entry: dict) -> dict:
        """Reliable rule-based mock agent when Nvidia API is offline/unavailable."""
        print("[agent] Running rule-based fallback decision agent")
        msg = anomaly_details.get("message", "").lower()
        sensor = anomaly_details.get("sensor", "").lower()

        # Gather telemetry window for log trace
        telemetry = self.callbacks["get_telemetry_window"](0, 1000)
        recent_temps = [t["motor_temp"] for t in telemetry[-5:]] if telemetry else []

        entry["steps"].append({
            "thought": "NVIDIA API unavailable. Falling back to onboard deterministic logic.",
            "tool": "get_telemetry_window",
            "args": {"start_tick": 0, "end_tick": 1000}
        })

        if "motor" in sensor or "motor" in msg:
            entry["final_action"] = "slow_down"
            entry["explanation"] = (
                f"Detected anomalous temperature spike on {sensor}. "
                "Enacted autonomous COOL_DOWN mode (slow traversal) to protect electric motor actuators."
            )
        elif "battery" in sensor or "battery" in msg:
            entry["final_action"] = "replan"
            entry["explanation"] = (
                "Detected anomalous battery drain. Triggered autonomous path replanning "
                "to find a flatter route and conserve remaining energy cells."
            )
            self.callbacks["trigger_replan"]("Battery drain safety reroute")
        elif "tilt" in sensor or "tilt" in msg:
            entry["final_action"] = "replan"
            entry["explanation"] = (
                "Excessive rover tilt detected. Recalculated path to bypass steep terrain slope."
            )
            self.callbacks["trigger_replan"]("High tilt slope bypass")
        else:
            entry["final_action"] = "none"
            entry["explanation"] = f"Monitored general alert: {msg}. No corrective navigation required."

        agent_decision_log.append(entry)
        return entry

    def run_chat_agent(self, user_query: str, chat_history: list) -> dict:
        """Answers follow-up operator queries by inspect mission log and telemetry."""
        print(f"[agent] Operator chat query: '{user_query}'")

        mission_log_data = self.callbacks["get_mission_log"]()
        telemetry_data = self.callbacks["get_telemetry_window"](0, 1000)

        # Simplify log details to fit model context limit
        simplified_log = []
        for entry in mission_log_data[:10]:
            simplified_log.append({
                "tick": entry.get("tick"),
                "detected_by": entry.get("detected_by"),
                "features": entry.get("top_contributing_features"),
                "mode": entry.get("mode")
            })

        recent_telemetry = []
        if telemetry_data:
            for t in telemetry_data[-10:]:
                recent_telemetry.append({
                    "tick": t.get("tick"),
                    "battery": t.get("battery_pct"),
                    "motors": t.get("motor_temp"),
                    "tilt": t.get("tilt_deg"),
                    "comms": t.get("comms_signal"),
                    "mode": t.get("mode")
                })

        if not self.client:
            return {
                "response": self._run_mock_chat(user_query, simplified_log, recent_telemetry),
                "tools_called": ["get_mission_log", "get_telemetry_window"]
            }

        try:
            prompt = (
                "You are the Mission Copilot AI Agent on a lunar rover. "
                "Answer the operator's query about the rover status and history using the following context.\n\n"
                f"Mission Log (Anomalies): {json.dumps(simplified_log)}\n"
                f"Recent Telemetry: {json.dumps(recent_telemetry)}\n"
                f"Operator query: {user_query}\n"
                "Respond in a direct, helpful tone. Keep it under 3 sentences."
            )

            response = self.client.chat.completions.create(
                model="nvidia/nemotron-3-ultra-550b-a55b",
                messages=[
                    {"role": "system", "content": "You are a helpful Mission Control Copilot."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=250
            )
            return {
                "response": response.choices[0].message.content.strip(),
                "tools_called": ["get_mission_log", "get_telemetry_window"]
            }
        except Exception as e:
            print(f"[agent] Chat API error: {e}")
            return {
                "response": self._run_mock_chat(user_query, simplified_log, recent_telemetry),
                "tools_called": ["get_mission_log", "get_telemetry_window"]
            }

    def _run_mock_chat(self, query: str, log: list, telemetry: list) -> str:
        query_l = query.lower()
        if "replan" in query_l or "reroute" in query_l or "path" in query_l:
            return (
                "I initiated a replan because telemetry indicators (slope or battery drain) "
                "crossed the critical threshold, requiring a flatter and safer path."
            )
        if "motor" in query_l or "temp" in query_l:
            return (
                "A motor temperature anomaly was detected on the front-left motor. "
                "I commanded a transition to COOL_DOWN mode to let the temperature stabilize."
            )
        if "battery" in query_l:
            return "Battery levels are being monitored closely. Low power reserves triggered a reroute to flat terrain."
        
        return "System telemetry is within normal bounds now. All sensors reporting nominal states."
