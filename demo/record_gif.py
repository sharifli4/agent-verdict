"""
Generate an explanatory demo GIF that shows:
  1. The code you write
  2. What each pipeline stage does
  3. The final verdict
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# --- config ---
WIDTH = 1060
LINE_HEIGHT = 22
PADDING = 24
FONT_SIZE = 16
BG = (24, 24, 27)
FG = (212, 212, 216)
GREEN = (74, 222, 128)
RED = (248, 113, 113)
YELLOW = (250, 204, 21)
CYAN = (103, 232, 249)
DIM = (113, 113, 122)
PURPLE = (167, 139, 250)
ORANGE = (251, 146, 60)
WHITE = (255, 255, 255)
SEPARATOR_COLOR = (63, 63, 70)

try:
    font = ImageFont.truetype("/usr/share/fonts/TTF/JetBrainsMonoNerdFont-Regular.ttf", FONT_SIZE)
except OSError:
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", FONT_SIZE)
    except OSError:
        font = ImageFont.load_default()

try:
    font_bold = ImageFont.truetype("/usr/share/fonts/TTF/JetBrainsMonoNerdFont-Bold.ttf", FONT_SIZE)
except OSError:
    font_bold = font


# --- colored line type ---
Segment = tuple[str, tuple[int, int, int]]
ColoredLine = list[Segment]


def seg(text: str, color: tuple[int, int, int] = FG) -> Segment:
    return (text, color)


def render_frame(colored_lines: list[ColoredLine], height: int | None = None) -> Image.Image:
    h = height or (PADDING * 2 + len(colored_lines) * LINE_HEIGHT + 10)
    img = Image.new("RGB", (WIDTH, h), BG)
    draw = ImageDraw.Draw(img)
    y = PADDING
    for segments in colored_lines:
        x = PADDING
        for text, color in segments:
            f = font_bold if color == WHITE else font
            draw.text((x, y), text, fill=color, font=f)
            bbox = f.getbbox(text)
            x += bbox[2] - bbox[0]
        y += LINE_HEIGHT
    return img


def blank_line() -> ColoredLine:
    return [seg("")]


def title_line(text: str) -> ColoredLine:
    return [seg(f"  {text}  ", CYAN)]


def separator() -> ColoredLine:
    return [seg("─" * 80, SEPARATOR_COLOR)]


def comment(text: str) -> ColoredLine:
    return [seg(f"# {text}", DIM)]


def code_line(text: str) -> ColoredLine:
    """Simple syntax-ish coloring."""
    segments: list[Segment] = []
    keywords = ["from", "import", "def", "return", "await", "async"]
    decorators_started = text.strip().startswith("@")

    if decorators_started:
        segments.append(seg(text, ORANGE))
    else:
        words = text.split(" ")
        for i, w in enumerate(words):
            if i > 0:
                segments.append(seg(" "))
            if w in keywords:
                segments.append(seg(w, PURPLE))
            elif w.startswith('"') or w.startswith("'"):
                segments.append(seg(w, GREEN))
            elif w.startswith("#"):
                segments.append(seg(" ".join(words[i:]), DIM))
                break
            else:
                segments.append(seg(w, FG))
    return segments


def label_value(label: str, value: str, val_color: tuple[int, int, int] = FG) -> ColoredLine:
    return [seg(f"  {label:18s}", DIM), seg(value, val_color)]


def stage_header(name: str, desc: str) -> list[ColoredLine]:
    return [
        blank_line(),
        [seg(f"  [{name}]", ORANGE), seg(f"  {desc}", DIM)],
    ]


def hold_frames(
    frames: list[Image.Image],
    durations: list[int],
    colored_lines: list[ColoredLine],
    duration: int,
    height: int,
):
    frames.append(render_frame(colored_lines, height))
    durations.append(duration)


def reveal_lines(
    frames: list[Image.Image],
    durations: list[int],
    base: list[ColoredLine],
    new_lines: list[ColoredLine],
    per_line_ms: int,
    final_hold_ms: int,
    height: int,
):
    for i, line in enumerate(new_lines):
        current = base + new_lines[: i + 1]
        dur = final_hold_ms if i == len(new_lines) - 1 else per_line_ms
        hold_frames(frames, durations, current, dur, height)
    return base + new_lines


def main():
    project_dir = Path(__file__).parent.parent
    out_path = project_dir / "demo" / "demo.gif"

    frames: list[Image.Image] = []
    durations: list[int] = []
    H = 580  # fixed height for all frames

    # =========================================================
    # SCENE 1: Show the concept
    # =========================================================
    intro = [
        blank_line(),
        [seg("  agent-verdict", WHITE)],
        blank_line(),
        [seg("  Your agent gives you an answer.", FG)],
        [seg("  But is it actually correct?", FG)],
        blank_line(),
        [seg("  This library checks it in 3 steps:", DIM)],
        blank_line(),
        [seg("  1. ", DIM), seg("Confidence", CYAN), seg("   — is the agent even sure about this?", DIM)],
        [seg("  2. ", DIM), seg("Verification", CYAN), seg(" — would a fresh reviewer agree?", DIM)],
        [seg("  3. ", DIM), seg("Adversarial", CYAN), seg("  — can it survive a counter-argument?", DIM)],
        blank_line(),
        [seg("  If the answer fails any step, it gets dropped.", YELLOW)],
    ]
    reveal_lines(frames, durations, [], intro, 200, 4000, H)

    # =========================================================
    # SCENE 2: Show the code
    # =========================================================
    code_header = [
        separator(),
        title_line("Step 1: Write your agent, add @verdict"),
        separator(),
        blank_line(),
    ]
    hold_frames(frames, durations, code_header, 1000, H)

    code = [
        code_line('from agent_verdict import verdict'),
        code_line('from agent_verdict.llm.anthropic import AnthropicProvider'),
        blank_line(),
        code_line('llm = AnthropicProvider()'),
        blank_line(),
        code_line('@verdict(llm=llm, task_context="Find security bugs")'),
        code_line('def analyze(code: str) -> str:'),
        code_line('    return my_agent_logic(code)'),
        blank_line(),
        comment("That's it. Now analyze() returns a Verdict, not a string."),
        comment("If the answer is bad, it raises DroppedResultError."),
    ]
    base = reveal_lines(frames, durations, code_header, code, 150, 3500, H)

    # =========================================================
    # SCENE 3: Run it — good result
    # =========================================================
    run_header = [
        separator(),
        title_line("Step 2: Run it — what happens inside"),
        separator(),
        blank_line(),
        comment('Agent returns: "SQL injection found on line 14"'),
        comment("Now agent-verdict checks this answer..."),
    ]
    hold_frames(frames, durations, run_header, 2000, H)

    # Stage 1: Confidence
    s1 = stage_header("CONFIDENCE", "How sure is the LLM about this answer?")
    base = reveal_lines(frames, durations, run_header, s1, 150, 800, H)
    s1_results = [
        label_value("confidence:", "0.89", GREEN),
        label_value("relevance:", "0.88", GREEN),
        label_value("justification:", "SQL injection via unsanitized input in query", FG),
        [seg("  >>> ", DIM), seg("PASSED", GREEN), seg(" — confidence above threshold", DIM)],
    ]
    base = reveal_lines(frames, durations, base, s1_results, 200, 1500, H)

    # Stage 2: Verification
    s2 = stage_header("VERIFICATION", "Fresh look — would you reach the same answer?")
    base = reveal_lines(frames, durations, base, s2, 150, 800, H)
    s2_results = [
        label_value("verified:", "True", GREEN),
        label_value("reason:", "f-string passes user input directly into WHERE clause", FG),
        [seg("  >>> ", DIM), seg("PASSED", GREEN), seg(" — independently confirmed", DIM)],
    ]
    base = reveal_lines(frames, durations, base, s2_results, 200, 1500, H)

    # Stage 3: Adversarial
    s3 = stage_header("ADVERSARIAL", "Attack the answer, then try to defend it")
    base = reveal_lines(frames, durations, base, s3, 150, 800, H)
    s3_results = [
        label_value("counter-argument:", "Input might be sanitized upstream by middleware", YELLOW),
        label_value("defense:", "No middleware configured — raw input hits the query", GREEN),
        label_value("defended:", "True", GREEN),
        [seg("  >>> ", DIM), seg("PASSED", GREEN), seg(" — defense held up", DIM)],
    ]
    base = reveal_lines(frames, durations, base, s3_results, 200, 2000, H)

    # =========================================================
    # SCENE 4: Final verdict
    # =========================================================
    verdict_section = [
        blank_line(),
        separator(),
        title_line("Verdict: SURVIVED"),
        separator(),
        blank_line(),
        label_value("result:", "SQL injection found on line 14", CYAN),
        label_value("confidence:", "0.89", GREEN),
        label_value("defended:", "True", GREEN),
        label_value("dropped:", "False", GREEN),
        blank_line(),
        comment("The answer passed all 3 stages. You can trust it."),
    ]
    reveal_lines(frames, durations, [], verdict_section, 200, 3500, H)

    # =========================================================
    # SCENE 5: What if the answer is bad?
    # =========================================================
    bad_header = [
        separator(),
        title_line("What if the answer is bad?"),
        separator(),
        blank_line(),
        comment('Agent returns: "There might be some security issues"'),
        comment("Vague, no specifics. Let's see what happens..."),
    ]
    hold_frames(frames, durations, bad_header, 2000, H)

    bad_s1 = stage_header("CONFIDENCE", "How sure is the LLM about this answer?")
    base_bad = reveal_lines(frames, durations, bad_header, bad_s1, 150, 800, H)
    bad_results = [
        label_value("confidence:", "0.18", RED),
        label_value("relevance:", "0.22", RED),
        blank_line(),
        [seg("  >>> ", DIM), seg("DROPPED!", RED), seg("  Confidence 0.18 below threshold 0.5", DIM)],
        blank_line(),
        comment("Pipeline stops here. No point running verification or adversarial."),
        comment("Only 1 LLM call was made. Bad answers fail fast."),
    ]
    base_bad = reveal_lines(frames, durations, base_bad, bad_results, 200, 2500, H)

    # =========================================================
    # SCENE 6: catch the error
    # =========================================================
    catch_section = [
        blank_line(),
        separator(),
        title_line("In your code, you catch it like this:"),
        separator(),
        blank_line(),
        code_line("try:"),
        code_line("    result = analyze(user_code)"),
        code_line("except DroppedResultError as e:"),
        code_line('    print(e.verdict.drop_reason)'),
        blank_line(),
        [seg("  Output: ", DIM), seg('"Confidence 0.18 below threshold 0.5"', RED)],
        blank_line(),
        blank_line(),
        [seg("  pip install agent-verdict", PURPLE)],
        [seg("  github.com/sharifli4/agent-verdict", DIM)],
    ]
    reveal_lines(frames, durations, [], catch_section, 150, 5000, H)

    # --- normalize and save ---
    normalized = []
    for frame in frames:
        if frame.height < H:
            new_frame = Image.new("RGB", (WIDTH, H), BG)
            new_frame.paste(frame, (0, 0))
            normalized.append(new_frame)
        elif frame.height > H:
            normalized.append(frame.crop((0, 0, WIDTH, H)))
        else:
            normalized.append(frame)

    normalized[0].save(
        str(out_path),
        save_all=True,
        append_images=normalized[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    size_kb = out_path.stat().st_size // 1024
    print(f"Saved: {out_path}")
    print(f"Frames: {len(normalized)}, size: {size_kb}KB")


if __name__ == "__main__":
    main()
