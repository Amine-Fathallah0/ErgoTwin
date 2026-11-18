# ErgoTwin: camera module

This is the computer-vision part of **ErgoTwin**, our #SSTHack 2026 project against musculoskeletal disorders.

It watches a camera or a video and finds every worker's skeleton. It measures the posture of the back,
neck, shoulders and knees and rates each one **safe**, **caution** or **bad** with the REBA method.
It times how long each posture is held and **flags** a bad posture that lasts. A web page shows the
result live, and the dashboard and the ankle pod can read the same data as JSON.

![The app's page in skeleton-only mode, on a ceiling camera over a produce-sorting line](docs/page.jpg)

*The page in skeleton-only mode: no camera picture is shown, only skeletons. On the right: the summary, the
worker most at risk on a body map, and every worker sorted by risk. Alerts are under the video.*

## Where to start

| You work on | Read |
|---|---|
| Anything | [1. What it does](#1-what-it-does), [2. Quick start](#2-quick-start), [3. Using the page](#3-using-the-page) |
| The dashboard or UI | [5. Connecting to the app](#5-connecting-to-the-app) |
| The ankle pod | [5.6 The ankle pod](#56-the-ankle-pod) |
| Ergonomics and validation | [6. How posture is judged](#6-how-posture-is-judged) |
| The story and the pitch | [1.2 Privacy](#12-privacy), [11. Honest limits](#11-honest-limits) |

**Contents**

1. [What it does](#1-what-it-does)
2. [Quick start](#2-quick-start)
3. [Using the page](#3-using-the-page)
4. [Video sources](#4-video-sources)
5. [Connecting to the app](#5-connecting-to-the-app)
6. [How posture is judged](#6-how-posture-is-judged)
7. [Settings](#7-settings)
8. [How the code is organised](#8-how-the-code-is-organised)
9. [Tests](#9-tests)
10. [Troubleshooting](#10-troubleshooting)
11. [Honest limits](#11-honest-limits)
12. [Not built yet](#12-not-built-yet)
13. [Credits and licences](#13-credits-and-licences)

---

## 1. What it does

```mermaid
flowchart LR
    A["Camera, video file<br/>or network stream"] --> B["Pose model on the GPU<br/>17 body points per person"]
    B --> C["Ceiling-camera<br/>correction"]
    C --> D["Joint angles"]
    D --> E["REBA: safe, caution or bad<br/>for each body region"]
    E --> F["Timers and flags"]
    F --> G["Web page"]
    F --> H["JSON: /state and /flags"]
    H --> I["Dashboard"]
    H --> J["Ankle pod"]
```

For every frame:

1. **Find people.** A YOLO pose model finds each person and 17 points on their body (shoulders, elbows,
   wrists, hips, knees, ankles and points on the face). A tracker follows each person from frame to frame,
   so the app knows who is who.
2. **Correct for the camera.** A ceiling camera looks down, so upright people seem to lean toward the
   edges of the picture. The app works out the camera's tilt from the workers and corrects for it
   ([details](#65-ceiling-cameras)).
3. **Measure angles.** Trunk bend, neck bend, upper-arm lift, elbow bend and knee bend.
4. **Rate each body region.** The angles become REBA scores. Each region is then **safe** (green),
   **caution** (amber) or **bad** (red).
5. **Time it.** Each region of each worker has a clock that counts how long its current level has lasted.
6. **Flag it.** A **bad** posture held for 5 seconds, or a **caution** posture held for 60 seconds, raises
   a flag. Each flag carries the pod's vibration code.

### 1.1 What you see

![Skeleton-only view of the warehouse clip](docs/skeleton-view.jpg)

*Skeleton-only view of a ceiling camera over a sorting line. In a crowd, workers who are fine carry only
their number. Workers who need attention carry their posture and how long they have held it.*

### 1.2 Privacy

These points are true of the code as it is now, and can go on the privacy slide:

- **Nothing is saved.** No video, picture or skeleton is written to disk. Flags are kept in memory only
  and are gone when the app stops.
- **Nothing leaves the laptop.** Everything runs on the laptop. The only download is the pose model, on
  the first start. The pose library's anonymous usage statistics are switched off inside the app.
- **Skeleton-only mode.** With **Skeleton only** ticked, the page and the video stream contain no camera
  pixels at all, only skeletons.
- **No identities.** There is no face recognition. Worker numbers are temporary: a number is given when
  someone appears and reused after they leave.

---

## 2. Quick start

Tested on Windows 11 with an NVIDIA RTX 3050 Laptop GPU (4 GB) and Python 3.13.

### 2.1 What you need

- [Git](https://git-scm.com/downloads)
- [Python](https://www.python.org/downloads/) 3.11 or newer
- An NVIDIA GPU with a recent driver. Without one it still works on the CPU, more slowly
  (see [2.5](#25-without-an-nvidia-gpu-or-on-mac-or-linux)).
- Optional: [uv](https://docs.astral.sh/uv/), a faster replacement for pip
- Optional: [ffmpeg](https://ffmpeg.org/), only to send a video as a live stream (see [4.4](#44-a-video-file-as-a-live-stream)).
  On Windows: `winget install Gyan.FFmpeg.Essentials`

### 2.2 Install

Open a terminal (PowerShell or Command Prompt) where you want the project:

```
git clone https://github.com/Amine-Fathallah0/ErgoTwin.git
cd ErgoTwin
```

Create a Python environment and install the packages. With **uv**:

```
uv venv .venv
uv pip install --python .venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

Or with plain **pip**:

```
python -m venv .venv
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\python -m pip install -r requirements.txt
```

The second line installs the GPU build of PyTorch (about 2.5 GB). Do it before the third line, or pip will
pick the CPU build.

### 2.3 Get a video

The `videos/` folder is empty after cloning, because clips are not stored in the repository. Either:

- Put any clip you have into `videos/`, or
- Make the synthetic test clip: `.venv\Scripts\python scripts\make_test_clip.py`, or
- Ask the team's computer-vision lead for the clips used in development: a warehouse ceiling camera and a
  3 × 3 grid recording of an assembly line.

### 2.4 Run

```
.\run.cmd
```

Open <http://localhost:8000>. Choose a clip and press **Start**.

The first start takes about 20 seconds while the pose model loads onto the GPU. On the very first start
it also downloads the model (about 50 MB), so be online then. When it runs, the top line of the page
shows the frame rate and the GPU's name. If it says "CPU", see [Troubleshooting](#10-troubleshooting).

Stop the app with **Ctrl+C** in the terminal.

### 2.5 Without an NVIDIA GPU, or on Mac or Linux

- Skip the PyTorch line of the install. pip then installs the CPU build.
- In `config.toml`, set `model = "yolo26n-pose.pt"` and `imgsz = 640` so it stays smooth on a CPU.
  It will still miss small, distant workers.
- On Mac or Linux the paths use `/`. Use `.venv/bin/python` instead of `.venv\Scripts\python`, and start
  the app with `.venv/bin/python -m ergotwin` instead of `run.cmd`.

---

## 3. Using the page

Everything fits on one screen, with no scrolling on a laptop. The page follows the computer's light or dark
setting.

| Part of the page | What it shows |
|---|---|
| **Top bar** | The status (**Idle**, **Loading model**, **Connecting**, **Live**, **Stream ended** or **Error**), the frame rate, the GPU and, for a ceiling camera, how far it looks down. Then the source picker (**Video file**, **Webcam** or **Stream URL**) with **Start** and **Stop**, the **View** picker for grid recordings ([4.6](#46-recordings-with-several-views-in-a-grid)), the **Skeleton only** switch, and **Settings**. |
| **Live view** | The video, with each worker's skeleton coloured by body region and a label with their number. In a crowd, workers who are fine carry only their number. |
| **Summary** | How many workers are in view, how many are bad, caution and safe, and how many are flagged right now. |
| **Most at risk** | The worker who most needs attention: a body map coloured by region, each region's posture and how long it has lasted, and the whole-body REBA score (1 to 15). Click a worker in the list to keep them here; click again or press **Esc** to let go. |
| **Workers** | One line per worker, most at risk first: their number (coloured by their worst region), that posture and how long it has lasted, one square each for **B**ack, **N**eck, **S**houlders and **K**nees, and the REBA score. Hover a square for its posture and time. |
| **Alerts** | Every alert, newest first: the time, the worker, the body region, what happened, and the pod's vibration pulses as dots. |
| **Settings** | **Seated work** (bent knees no longer count against the worker), and how long a bad or a caution posture must last before an alert. Changes apply at once. |

**Colours**: green = safe, amber = caution, red = bad, grey = that body part is not visible. A red frame
around a worker in the video means an alert is active for them. "Partial view" next to a REBA score means
a region was hidden and counted as neutral.

---

## 4. Video sources

### 4.1 Video file

Pick any clip in `videos/`. It plays at its own speed and **restarts at the end**, like a camera on the
same scene. The restart is treated as one continuous scene, so held times can grow longer than the clip.

### 4.2 Webcam

Choose **Webcam** and enter its number, usually `0`; try `1` for a second camera. *Not yet tested.*

### 4.3 IP camera or network stream

Choose **Stream URL** and enter its address, for example `rtsp://user:password@192.168.1.64:554/stream1`
(the exact path depends on the camera brand), or `udp://...` or `http://...`. Only UDP has been tested.

### 4.4 A video file as a live stream

This shows the app working on a live feed when you have no camera. It sends a file in a loop over the
network, the way an IP camera would. In a second terminal:

```
scripts\stream.cmd videos\warehouse_sorting.mp4
```

Then in the page choose **Stream URL**, enter `udp://127.0.0.1:5000` and press **Start**. Start the stream
first, then the app. A second number sets another port: `scripts\stream.cmd videos\clip.mp4 5001`.

On Mac or Linux, run ffmpeg directly:

```
ffmpeg -re -stream_loop -1 -i videos/clip.mp4 -an -c:v libx264 -preset ultrafast -tune zerolatency -g 30 -pix_fmt yuv420p -f mpegts "udp://127.0.0.1:5000?pkt_size=1316"
```

### 4.5 Getting a showcase clip

yt-dlp is installed with the project. Use footage you have the right to use:

```
.venv\Scripts\yt-dlp -f "bv*[vcodec^=avc1][height<=720]" -o "videos/demo.%(ext)s" <video address>
```

Good footage has a fixed camera and shows workers whole, from the side or at an angle, while they lift,
bend or reach.

To open a clip straight away: `.\run.cmd --source videos\demo.mp4`

In PowerShell, commands in the current folder need the `.\` in front (`.\run.cmd`). In Command Prompt it
works with or without it.

### 4.6 Recordings with several views in a grid

Some recordings show several camera views at once, for example a 3 × 3 grid. In the whole picture every
worker is tiny: on the 3 × 3 test clip the model found only 2 to 4 people per frame, mostly people walking past.

The **View** picker has two ways round this. Tiles are counted left to right, then top to bottom.

| Choice | What it does | On the 3 × 3 test clip |
|---|---|---|
| *3×3 · all tiles* | Cuts out all nine views, enlarges each one, and analyses them together. Everyone is drawn back on the whole grid. | 60 to 90 workers at about 4.5 fps |
| *3×3 · tile 5* (or any tile) | Cuts out that one view and enlarges it to the full picture. | about 8 workers at 20 fps |

With all tiles, each view is treated as a camera of its own: its own tracks, its own camera-tilt
correction, its own clocks. Worker numbers are shared, so no two workers on screen have the same number,
and the worker list and the alerts say which tile each worker is in (**T5**, *Tile 5 ·*).

All tiles uses a smaller, faster model than one tile (`yolo26s-pose.pt` instead of `yolo26m-pose.pt`),
because the model has nine pictures to look at in every frame. It is loaded the first time you choose all
tiles, which takes a few seconds. Both models together use about 1.8 GB of the graphics card's memory. Its
settings are under `[grid]` in `config.toml`:

- `tile_width = 1088`: the width each tile is enlarged to. At 960 the model finds about half the people,
  so do not go lower.
- `conf = 0.25`: how sure the model must be before it follows someone. It is lower than the 0.35 used
  elsewhere, because people in a grid are small and the model is less sure of them. Raise it if you see
  skeletons on things that are not people.

For seated work, such as assembly benches, also tick **Seated work** in **Settings**. Otherwise bent knees
under the bench count as a risk.

To start on a grid view, set `tile = "3x3:all"` or `tile = "3x3:5"` under `[camera]` in `config.toml`, or
send `{"tile": "3x3:all"}` to `POST /settings`.

---

## 5. Connecting to the app

Everything the page shows is also available as JSON, so the dashboard, the pod or any script can use it.
FastAPI also gives interactive documentation of every call at <http://localhost:8000/docs>.

### 5.1 Running it for the rest of the team

By default the app only accepts connections from the laptop it runs on. To let other laptops or the
pod reach it:

```
.\run.cmd --host 0.0.0.0
```

Others then use `http://<this laptop's address>:8000`. Find the address with `ipconfig` (the IPv4 address
of the Wi-Fi adapter). Windows Firewall asks the first time: allow Python on private networks. Venue Wi-Fi
often blocks devices from reaching each other. If so, put everyone on a phone's hotspot.

Another port: `.\run.cmd --port 8001`.

### 5.2 All the calls

| Call | What it does |
|---|---|
| `GET /state` | Everything: status, every worker in view, the latest 50 flags. |
| `GET /flags?after=N` | Only the flags numbered above N, oldest first. Made for the pod. |
| `GET /video` | The annotated live picture as an MJPEG stream. Works in an `<img>` tag. |
| `GET /videos` | The clips found in `videos/`. |
| `POST /start` | Body `{"source": "videos/demo.mp4"}`: start a clip, a webcam number (`"0"`) or a stream URL. |
| `POST /stop` | Stop. |
| `POST /settings` | Body with any of `privacy`, `seated` (true or false), `bad_hold_s`, `caution_hold_s` (seconds), `tile` (`"whole"`, a grid view such as `"3x3:5"`, or every tile with `"3x3:all"`). |
| `GET /` | The web page. |
| `GET /docs` | Interactive documentation of all of the above. |

Any web page may call these: cross-origin requests are allowed, so a dashboard on another port or opened
as a file works too.

### 5.3 `/state`

An example while running, shortened to one worker and one region:

```json
{
  "status": "running",
  "message": "",
  "source": "videos/warehouse_sorting.mp4",
  "fps": 15.6,
  "device": "NVIDIA GeForce RTX 3050 Laptop GPU",
  "model": "yolo26m-pose.pt",
  "privacy": false,
  "seated": false,
  "tile": "whole",
  "time": 41.6,
  "bad_hold_s": 5.0,
  "caution_hold_s": 60.0,
  "camera": {"learned": true, "tilt_deg": 35},
  "workers": [
    {
      "id": 7,
      "level": "caution",
      "posture": "Arm raised 69°",
      "held_s": 5.3,
      "flagged": false,
      "view": "front",
      "tile": null,
      "reba": {"score": 2, "risk": "low", "trunk": 1, "neck": 1, "legs": 1, "upper_arm": 3,
               "lower_arm": 2, "wrist": 1, "activity": 0, "complete": false},
      "regions": {
        "shoulders": {"level": "caution", "label": "Arm raised 69°", "angle": 69,
                      "held_s": 5.3, "flagged": false, "pulses": 1}
      }
    }
  ],
  "flags": [
    {"seq": 12, "t": 59.8, "clock": "20:41:07", "worker": 7, "region": "shoulders", "level": "caution",
     "angle": 69, "held_s": 5.0, "pulses": 1, "message": "Worker 7: arm raised 69° for 5 s", "tile": null}
  ]
}
```

**Top level**

| Field | Meaning |
|---|---|
| `status` | `idle`, `loading`, `connecting`, `running`, `ended` (a stream stopped) or `error`. |
| `message` | Why, when the status is not `running`. Empty while running. |
| `source` | What is playing. |
| `fps` | Frames processed per second. |
| `device`, `model` | The GPU (or `CPU`) and the pose model file. |
| `privacy`, `seated` | The **Skeleton only** and **Seated work** settings. |
| `tile` | `"whole"`, or the view of a grid recording being analysed, such as `"3x3:5"` or `"3x3:all"`. In a worker or a flag, `tile` is the tile number (1 to 9 in a 3 × 3 grid), or `null` outside the all-tiles view. |
| `time` | Seconds since **Start**. |
| `bad_hold_s`, `caution_hold_s` | The current flag times in seconds. |
| `camera` | `learned`: whether the camera's tilt is known yet. `tilt_deg`: how far it looks down (0 = level). |
| `workers` | One entry per worker in view. |
| `flags` | The latest 50 flags, newest first. |

**Each worker**

| Field | Meaning |
|---|---|
| `id` | The worker's number while in view. It is reused after they leave, so it is not an identity. |
| `level` | The worst body region: `safe`, `caution`, `bad`, or `null` when too little is visible. |
| `posture` | Words for that region, such as `"Bent forward 65°"`. `"Neutral posture"` when safe. |
| `held_s` | How long that posture has lasted, in seconds. 0 when safe. |
| `flagged` | `true` while any region is past its flag time. |
| `view` | How the camera sees the worker: `side`, `oblique` or `front`. Angles are best from the side. |
| `reba` | The whole-body REBA score: `score` (1 to 15), `risk`, the part scores, `activity`, and `complete` (`false` if a region was hidden and counted as neutral). |
| `regions` | `back`, `neck`, `shoulders`, `knees`, each with the fields below. |

**Each region**

| Field | Meaning |
|---|---|
| `level` | `safe`, `caution`, `bad`, or `null` when not visible. |
| `label` | Words for the posture, such as `"Knees bent 45°"` or `"Not visible"`. |
| `angle` | The main angle in degrees, or `null`. |
| `held_s` | How long this region has been at this level, in seconds. |
| `flagged` | `true` while this region is past its flag time. |
| `pulses` | The pod's vibration code for this region (see below). |

**Each flag**

| Field | Meaning |
|---|---|
| `seq` | The flag's number. It goes up by 1 with every flag for as long as the app runs, across Stop and Start. |
| `t`, `clock` | Seconds since Start, and the time of day. |
| `worker`, `region`, `level` | Who, which body region, and `caution` or `bad`. |
| `angle`, `held_s` | The angle and how long the posture had been held when the flag fired. |
| `pulses` | The pod's vibration code. |
| `message` | A sentence ready to show, such as `"Worker 7: arm raised 69° for 5 s"`. |

**Vibration codes**, from the project brief: **1** pulse = arms and shoulders, **2** = trunk and back (the
neck uses 2 as well), **3** = legs and knees.

### 5.4 `/flags`

`GET /flags?after=12` returns the flags numbered above 12, oldest first, and the newest number:

```json
{"latest": 14, "flags": [{"seq": 13, "...": "..."}, {"seq": 14, "...": "..."}]}
```

Remember the highest `seq` you have handled and ask for `after=` that number next time. If `latest` is
ever **lower** than your number, the app was restarted: start again from the current `latest`.

### 5.5 The dashboard

The live picture needs no code:

```html
<img src="http://localhost:8000/video" width="640" alt="ErgoTwin live view">
```

Read the data twice a second:

```html
<script>
  async function refresh() {
    const state = await (await fetch("http://localhost:8000/state")).json();
    for (const worker of state.workers) {
      console.log(worker.id, worker.level, worker.posture, worker.held_s);
    }
  }
  setInterval(refresh, 500);
</script>
```

Replace `localhost` with the camera laptop's address if the dashboard runs on another machine (see [5.1](#51-running-it-for-the-rest-of-the-team)).

### 5.6 The ankle pod

The pod asks `/flags` every half second and vibrates `pulses` times for each new flag. Below is a
**starting point for an ESP32 that has not been tried on hardware**. It needs the ArduinoJson library
(version 7), and the app must be started with `.\run.cmd --host 0.0.0.0`.

```cpp
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

const char* WIFI_NAME = "your-network";
const char* WIFI_PASSWORD = "your-password";
const String APP = "http://192.168.1.20:8000";  // the laptop running ErgoTwin
const int MOTOR_PIN = 25;                       // drives the vibration motor through a transistor

long lastSeq = 0;      // number of the last flag handled
bool started = false;  // the first answer only tells us where to start

void buzz(int pulses) {
  for (int i = 0; i < pulses; i++) {
    digitalWrite(MOTOR_PIN, HIGH);
    delay(250);
    digitalWrite(MOTOR_PIN, LOW);
    delay(250);
  }
}

void setup() {
  pinMode(MOTOR_PIN, OUTPUT);
  WiFi.begin(WIFI_NAME, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) delay(250);
}

void loop() {
  HTTPClient http;
  http.begin(APP + "/flags?after=" + String(lastSeq));
  if (http.GET() == 200) {
    JsonDocument doc;
    if (!deserializeJson(doc, http.getString())) {
      long latest = doc["latest"].as<long>();
      if (!started || latest < lastSeq) {  // just switched on, or the app restarted: skip old flags
        lastSeq = latest;
        started = true;
      } else {
        for (JsonObject flag : doc["flags"].as<JsonArray>()) {
          lastSeq = flag["seq"].as<long>();
          buzz(flag["pulses"].as<int>());
          delay(600);  // a gap between two flags
        }
      }
    }
  }
  http.end();
  delay(500);
}
```

The pod reacts to **every** worker's flags. Pairing one pod with one skeleton (the Bluetooth idea in the
brief) is not built. For the demo, keep one person in front of the camera, or check `flag["worker"]`.

### 5.7 Changing settings while it runs

The boxes on the page do this. From PowerShell:

```
Invoke-RestMethod -Method Post -Uri http://localhost:8000/settings -ContentType application/json -Body '{"caution_hold_s": 10}'
```

For a demo, a short caution time makes flags appear quickly. Settings changed this way last until the app
stops. To keep them, edit `config.toml`.

---

## 6. How posture is judged

This part is for checking the method. Every number below can be changed in `config.toml`.

### 6.1 The angles

All angles come from 2D points in the picture, after the ceiling-camera correction.

| Angle | How it is measured | Notes |
|---|---|---|
| **Trunk bend** | The line from mid-hip to mid-shoulder, against vertical. Forward is positive. | Leaning back is reported only from a clean side view. Seen from the front, a tilt in the picture is a **sideways bend**. A bend toward the camera is invisible in a flat picture; from a level camera with the legs in view it is estimated from how short the torso looks, from 25° up. |
| **Neck bend** | The line from shoulders to ears, against the trunk line, minus 10° (in a neutral neck the ear sits a little ahead of the shoulder). | Measured only when the neck is at least 25 pixels long in the picture. From the front, only a sideways tilt is measured: the line through the ears against the line through the shoulders, so a head bowed over the work is not mistaken for a tilt. A head bowed toward the camera is not measured. |
| **Upper arm** | The line from shoulder to elbow, against the trunk line pointing down. 0° = arm hanging. | Arm behind the body is told apart only from a side view. Sideways lift (abduction) is counted only from a level camera facing the worker, above 45°. |
| **Elbow** | The bend at the elbow. 0° = straight arm. | Feeds the REBA lower-arm score: 60 to 100° is best. |
| **Knee** | The bend at the knee. 0° = straight leg. | The **straighter** knee counts, so walking (one knee bent at a time) is not a crouch. |

Angles are smoothed over a few frames to calm jitter.

### 6.2 From angles to safe, caution and bad

Each region's colour comes from its REBA part score:

| Region | REBA part score | Safe | Caution | Bad |
|---|---|---|---|---|
| **Back** | Trunk | 1–2: upright, bent under 20°, or leaning back under 20° | 3: bent 20–60°, leaning back over 20°, or bent under 20° and sideways | 4+: bent over 60°, or bent 20–60° and sideways |
| **Neck** | Neck | 1: bent under 20° | 2: bent over 20°, tilted back over 15°, or tilted sideways over 25° | 3: bent or tilted back, and tilted sideways |
| **Shoulders** | Upper arm | 1–2: arm under 45° | 3: raised 45–90° | 4+: above shoulder height (over 90°) |
| **Knees** | Legs | 1: both knees under 30° | 2: both bent 30–60° | 3: both bent over 60° |

"Sideways" means over 15° for the trunk and over 25° for the neck. The neck is short in the picture, so
its readings are noisier. With **Seated work** on, the knees always count as safe.

### 6.3 The whole-body REBA score

The app follows the published method: Table A combines trunk, neck and legs, and the load is added. Table B
combines upper arm, lower arm and wrist, and the grip is added. Table C combines the two, and the activity
points are added. The result runs from 1 to 15:

| Score | Risk |
|---|---|
| 1 | Negligible |
| 2–3 | Low |
| 4–7 | Medium |
| 8–10 | High |
| 11–15 | Very high |

The camera supplies the trunk, neck, legs, upper-arm and lower-arm scores. The rest are **station settings**
in `config.toml`, because a camera cannot see them: load weight, shock, grip quality, wrist posture, arm
support, repeated small actions, and rapid posture changes. One activity point is automatic: **+1 when a
caution or bad posture has been held for over a minute**.

The worker's **badge** is the worst region, not this score. A worker can be "bad" (arm above the shoulder)
with a low whole-body score, because the other regions are fine and no load is set.

### 6.4 Time and flags

| Rule | Default |
|---|---|
| A new level counts only once it has lasted this long (it then counts from when it first appeared) | 0.6 s |
| A region hidden for less than this keeps its clock running | 1 s |
| Flag a **bad** posture held this long | 5 s |
| Flag a **caution** posture held this long | 60 s |
| After a flag, the same region of the same worker is not flagged again for | 20 s |
| A worker is shown once followed for | 1 s |
| A worker not seen for this long is forgotten (their clocks stop) | 3 s |

### 6.5 Ceiling cameras

A camera that looks down shows upright people leaning away from a point below the picture, more and more
toward the edges, and shortened from head to foot. Read naively, everyone near the edges would look bent.

The app collects the lines of straight legs (or trunks, when the legs are hidden) of the workers it sees.
It finds the point they all point away from, and from that point works out the camera's tilt. Each
skeleton is then turned and stretched back to what a level camera would show before anything is measured.
This takes a few seconds and needs several workers spread across the picture. On the development clip it
found a tilt of about 35°.

The app accepts only a level camera or one that looks down, so a crowd leaning the same way is not mistaken
for a tilted camera. One case can still fool it: nearly everyone leaning **outward** from the centre of the
picture the whole time, with legs hidden. If you know the tilt, set `tilt_deg` in `config.toml` and nothing
is learned.

### 6.6 Validating it

The brief's plan is to have an ergonomist score postures by hand with REBA and compare. What exists now:
`/state` gives every angle and part score live, and the page shows them. A CSV export of angles and scores
per frame, which would make the comparison quick, is [not built yet](#12-not-built-yet).

---

## 7. Settings

All settings live in [`config.toml`](config.toml), with a comment on every line. Restart the app after
editing it.

| Section | What it controls |
|---|---|
| `[pose]` | The pose model and its speed. `imgsz = 1280` finds small, distant workers in a wide shot; `640` is faster and enough for one person near the camera. Smaller models: `yolo26s-pose.pt`, `yolo26n-pose.pt`. |
| `[station]` | What the camera cannot see: load, grip, wrist posture, seated work. These change the whole-body REBA score. |
| `[angles]` | Thresholds for upright, sideways and abduction, how front and side views are told apart, the smallest person and neck that are measured, and smoothing. |
| `[camera]` | Learn the ceiling-camera tilt, or give it with `tilt_deg`; the lens's field of view; and `tile`, the view to analyse in a grid recording. |
| `[grid]` | The model, tile size and confidence used when every tile of a grid is analysed at once (see [4.6](#46-recordings-with-several-views-in-a-grid)). |
| `[timing]` | Flag times, cooldown, and how long workers are kept or ignored (see [6.4](#64-time-and-flags)). |
| `[levels]` | The REBA part score at which each region turns caution and bad (see [6.2](#62-from-angles-to-safe-caution-and-bad)). |

**Speed** on the RTX 3050 Laptop GPU: about 45 ms per frame for a wide shot with 17 people at `imgsz = 1280`,
so a 15 fps camera runs in real time.

---

## 8. How the code is organised

```
ErgoTwin/
├── run.cmd                  starts the app (Windows)
├── config.toml              every setting, with a comment on each line
├── requirements.txt         Python packages (install GPU PyTorch first, see 2.2)
├── pyproject.toml           project and test settings
├── ergotwin/                the app
│   ├── __main__.py          command line: python -m ergotwin [--source] [--host] [--port]
│   ├── server.py            web server: the page, the video stream, the JSON calls
│   ├── engine.py            background thread: reads frames, runs the pipeline, draws
│   ├── sources.py           webcam, video file or network stream, newest frame only
│   ├── pose.py              YOLO pose model on the GPU, with tracking
│   ├── pipeline.py          per frame: from skeletons to each worker's state
│   ├── camera.py            learns a ceiling camera's tilt and corrects skeletons
│   ├── angles.py            joint angles from body points
│   ├── reba.py              REBA tables and scoring
│   ├── posture.py           clocks, flags, and the words for each posture
│   ├── tiles.py             cuts the views out of a grid recording
│   ├── draw.py              skeletons and labels on the picture
│   ├── skeleton.py          the 17 body points
│   ├── config.py            every setting and its default
│   └── static/index.html    the web page
├── scripts/
│   ├── stream.cmd           sends a video file as a live UDP stream with ffmpeg
│   └── make_test_clip.py    builds a synthetic test clip
├── tests/                   automated tests (pytest)
├── docs/                    pictures for this README
├── videos/                  your clips (not stored in the repository)
└── models/                  pose model, downloaded on first start (not stored in the repository)
```

**One frame's journey:** `sources.py` keeps the newest frame. `engine.py` passes it to `pose.py` (people
and their points), then to `pipeline.py`. That runs `camera.py`, `angles.py`, `reba.py` and `posture.py`
for each person. `draw.py` paints the result, and `server.py` serves the picture and the state.

The pose model sits behind one small class in `pose.py`. Another model (MediaPipe, or one with hand points
for wrists) can replace it without touching the rest.

---

## 9. Tests

```
.venv\Scripts\python -m pytest
```

There are 173 tests, and they run in about two seconds without a GPU. They cover:

- the angle maths, on stick figures with known angles seen from the side and the front;
- the REBA bands and tables, which were checked cell by cell against the Ergonomics Plus worksheet;
- the ceiling-camera learning, on crowds drawn as tilted cameras would show them, including crowds that
  lean to try to fool it;
- the clocks, flags, flag numbers and pod codes;
- cutting the views out of a grid recording, and drawing each tile's workers back in the right place,
  with numbers shared across tiles;
- the pipeline from skeletons to the JSON a worker gets.

They do not test the web page, the video sources or the pose model; those were checked by hand.

---

## 10. Troubleshooting

| Problem | What to do |
|---|---|
| The page says "The app is not running" | The terminal running `run.cmd` was closed or the app stopped. Start it again. |
| Stuck on "Loading model" on the first start | It is downloading the model (about 50 MB). Wait, and check the internet connection. |
| The top line says "CPU", or it runs at 2–3 fps | The CPU build of PyTorch is installed. Repeat the PyTorch line of [2.2](#22-install), then restart. |
| "Address already in use" | Something else uses port 8000: `.\run.cmd --port 8001`. |
| Stream URL stays on "Connecting" | Start the stream before pressing Start. Check the address and the port. |
| `'ffmpeg' is not recognized` | Install ffmpeg (see [2.1](#21-what-you-need)), then open a **new** terminal. |
| A teammate cannot open the page | Start with `.\run.cmd --host 0.0.0.0`, allow Python in Windows Firewall, and make sure you are on the same network. Try a phone hotspot. |
| The webcam does not open | Close other apps that use it (Teams, Zoom), or try number `1`. |
| A worker's number changes | They were hidden for over about 3 seconds and came back as a new worker. Their clocks restart. |
| No flags appear | Bad postures need 5 s, caution 60 s. Lower the times in the page for a demo. |
| `make_test_clip.py` fails | Run it from the project folder with the project's Python: `.venv\Scripts\python scripts\make_test_clip.py`. |

---

## 11. Honest limits

For the limits slide:

- **Not validated against an ergonomist yet.** It has run on one real clip, a ceiling camera over a
  produce-sorting line, where the readings look plausible. Angle errors against hand-scored REBA are still
  to be measured.
- **One flat camera.** Bends sideways to the camera are measured well. Bends toward or away from it are
  mostly not, so a worker facing the camera can read "Upright" while bending.
- **Ceiling cameras misread some arm positions.** An arm reaching toward the camera looks like an arm
  hanging down; an arm reaching away looks raised.
- **Small people.** Workers under 50 pixels tall are not measured, and necks only from 25 pixels. In a wide
  ceiling shot that usually means no neck reading.
- **Hidden legs.** Behind a bench or boxes the knees are not scored, and the whole-body score is marked partial.
- **Grid recordings.** Each view is small, so even enlarged, workers bowed low over the nearest bench are
  sometimes missed. All tiles at once runs at about 4.5 fps, enough for the clocks but not smooth to watch;
  pick one tile for a smooth picture.
- **No wrists.** The pose model has no hand points, so the REBA wrist score is a station setting.
- **Not detected:** standing on one leg, twisting, raised shoulders. Their REBA adjustments are not applied.
- **Numbers are not identities.** Someone hidden for a few seconds comes back as a new worker. On the
  development clip, about 12 of 15 workers kept their number for over 30 seconds.
- **Not tested:** webcam, a real IP camera, Mac and Linux, and the pod sketch.
- It supports an ergonomist's judgment and does not replace it.

---

## 12. Not built yet

- **CSV export** of angles and scores per frame, for the validation against hand-scored REBA.
- **Wrist angles**, which need a pose model with hand points.
- **Pairing a pod with a skeleton**, so each pod only vibrates for its own wearer.
- **Several cameras** watching the same station.
- The other parts of ErgoTwin (Body Budget, cascade, task rotation, what-if advisor) belong to other
  modules. This one gives them their inputs: per-worker, per-region levels, angles and times.

---

## 13. Credits and licences

- **REBA**: Hignett, S. and McAtamney, L. (2000). *Rapid Entire Body Assessment (REBA)*. Applied Ergonomics
  31(2), 201–205. The tables were checked against the [Ergonomics Plus REBA worksheet](https://ergo-plus.com/wp-content/uploads/REBA.pdf).
- **Pose model**: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), licensed under AGPL-3.0.
  AGPL asks that software built on it and offered to others, including over a network, is published under
  the same licence. A commercial version would need an Ultralytics licence or another pose model; only
  `ergotwin/pose.py` would change.
- No licence has been chosen for this repository yet.
