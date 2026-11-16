"""Define the geometry constants, the column grid, and the scoped stylesheet of every notebook control panel.

Responsibilities:
  - Left-align the text of the buttons and checkboxes in popovers
  - Fix the pixel geometry of rows, gaps, buttons, status line, plot gap, popovers, and tab headers
  - Estimate text widths generously in ch so that labels and options are never cut
  - Provide the column grid on which every element of a panel starts and ends
  - Render the stylesheet scoped below the root class, and carry it as a zero-height HTML widget

Design principles:
  - One owner for every number of the control standard; constructors and panels only read these constants
  - Every rule is scoped below the root class, so notebooks without the standard keep their look
  - Elements keep their width and rows never wrap; a row wider than the output scrolls inside itself

This module does NOT:
  - Construct controls, rows, or panels
  - Decide which control a view uses or how a selection is resolved
"""

import math
from dataclasses import dataclass
from typing import Final

import ipywidgets as widgets

LINE_PX: Final = 28  # height of every control and of one row's control line
ROW_GAP_PX: Final = 6  # vertical gap below a row's control line (a scrolling row shows its scrollbar there)
ROW_PITCH_PX: Final = LINE_PX + ROW_GAP_PX  # height reserved per panel row
GAP_PX: Final = 16  # an element to the next label
LABEL_GAP_PX: Final = 2  # a label to its own control
STEP_GAP_PX: Final = 2  # a number field to its step buttons and between the buttons
BUTTON_PX: Final = 34  # width of every step button
STATUS_PX: Final = 20  # one reserved status line
PLOT_GAP_PX: Final = 20  # status line to the plot below
TRACK_PX: Final = 40  # one column track of the grid; the column pitch is TRACK_PX + GAP_PX
CH_PX: Final = 8  # one ch of the widget font in px, slightly generous
TAB_HEIGHT_PX: Final = LINE_PX  # tab header height, tall enough that the bar never shows scroll arrows
TAB_PADDING_CH: Final = 3  # tab header padding plus a margin, in ch
TAB_WIDTH_FACTOR: Final = 0.875  # the generous estimate left headers about 1/8 too wide on screen
TAB_WIDTH_CH_MAX: Final = 100  # the stylesheet carries one width rule per tab width up to this many ch
RADIO_GAP_PX: Final = 24  # two stacked radio groups side by side outside panels (a quarter less than the former 32 px)
FIELD_ALLOWANCE_CH: Final = 3  # number field padding, border, and margin around its left-aligned value
BUTTON_ALLOWANCE_CH: Final = 4  # button padding around its text
DROPDOWN_ALLOWANCE_CH: Final = 5  # dropdown padding plus its arrow
ROOT_CLASS: Final = "glp-ui"

_UPPER_CH: Final = 1.25  # an upper-case letter is wider than the digit the ch unit measures
_WIDE_CH: Final = 1.5  # m, w, M, W and @
_WIDE_CHARACTERS: Final = frozenset("mwMW@")

_STYLESHEET: Final = """
.glp-ui {
  --jp-widgets-margin: 0px;
  --jp-widgets-inline-height: @LINE@px;
  --jp-widgets-container-padding: 0px;
  --jp-widgets-horizontal-tab-height: @TABH@px;
}
.glp-ui .widget-inline-hbox .widget-label, .glp-ui .jupyter-widget-inline-hbox .jupyter-widget-label {
  text-align: left; margin-right: @LABELGAP@px;
}
.glp-ui .glp-row {
  display: flex; flex-flow: row nowrap; align-items: center; gap: 0 @GAP@px;
  min-height: @LINE@px; margin: 0; padding: 0; overflow-x: auto; overflow-y: hidden;
}
.glp-ui .glp-row::-webkit-scrollbar { height: @ROWGAP@px; }
.glp-ui .glp-row::-webkit-scrollbar-thumb { background: rgba(128, 128, 128, 0.55); border-radius: 3px; }
.glp-ui .glp-rows { flex: 0 0 auto; }
.glp-ui .glp-rows > .glp-row { box-sizing: border-box; height: @PITCH@px; align-items: flex-start; }
.glp-ui .glp-row > *, .glp-ui .glp-step > *, .glp-ui .glp-checks > * { flex: 0 0 auto; }
.glp-ui .glp-step { display: flex; flex-flow: row nowrap; align-items: center; gap: @STEPGAP@px; }
.glp-ui .glp-checks { display: flex; flex-flow: row nowrap; align-items: center; gap: 0 @GAP@px; }
.glp-ui .glp-radio + .glp-radio { margin-left: calc(@RADIOGAP@px - @GAP@px); }
.glp-ui input[type=number]::-webkit-outer-spin-button,
.glp-ui input[type=number]::-webkit-inner-spin-button { -webkit-appearance: none; margin: 0; }
.glp-ui input[type=number] { -moz-appearance: textfield; appearance: textfield; text-align: left; }
.glp-ui .widget-dropdown > select, .glp-ui .jupyter-widget-dropdown > select { flex: 1 1 auto; min-width: 0; }
.glp-ui .widget-dropdown select, .glp-ui .widget-text input, .glp-ui .widget-button, .glp-ui .widget-toggle-button { height: @LINE@px; }
.glp-ui .glp-plot .widget-html-content, .glp-ui .glp-plot .jupyter-widget-html-content { line-height: normal; }
.glp-ui .glp-status { min-height: @STATUS@px; line-height: @STATUS@px; margin: 0 0 @PLOTGAP@px 0; overflow: visible; }
.glp-ui .glp-status .widget-html-content {
  line-height: @STATUS@px; overflow: visible; white-space: normal; overflow-wrap: anywhere; min-width: 0;
}
.glp-ui .glp-status-warning .widget-html-content { color: var(--jp-warn-color0, #b26a00); }
.glp-ui .glp-status-error .widget-html-content { color: var(--jp-error-color1, #c0392b); }
.glp-ui .glp-status progress { vertical-align: middle; width: 120px; height: 10px; margin-left: 8px; }
.glp-ui .glp-failure p { margin: 0 0 4px 0; color: var(--jp-error-color1, #c0392b); }
.glp-ui .glp-failure pre { max-height: 180px; overflow: auto; font-size: 11px; margin: 4px 0 0 0; }
.glp-ui.glp-panel, .glp-ui.glp-host, .glp-ui.glp-shell { overflow: visible; }
.glp-ui .glp-popover {
  position: relative; z-index: 20; align-self: start; justify-self: start; box-sizing: border-box; max-width: 100%;
  overflow-x: auto; overflow-y: hidden;
  padding: 6px 8px 6px 0; background: var(--jp-layout-color1, #fff);
  border: 1px solid var(--jp-border-color1, #888); border-left: none; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
}
.glp-ui .glp-popover .glp-row { overflow: visible; height: auto; }
.glp-ui .glp-popover .widget-toggle-button, .glp-ui .glp-popover .jupyter-button, .glp-ui .glp-popover .widget-checkbox {
  justify-content: flex-start; text-align: left;
}
.glp-ui .glp-plot { overflow-x: auto; overflow-y: hidden; }
.glp-ui .glp-plot img, .glp-ui .widget-image, .glp-ui .jupyter-widget-image { max-width: none; }
.glp-ui .glp-text pre { margin: 0; }
.glp-ui .lm-TabBar-content, .glp-ui .p-TabBar-content {
  display: flex !important; flex-flow: row nowrap !important; flex: 0 1 auto !important; min-width: 0 !important;
  max-width: 100%; height: auto !important; overflow-x: auto; overflow-y: hidden;
}
.glp-ui .lm-TabBar-tab, .glp-ui .p-TabBar-tab {
  flex: 0 0 auto !important; min-width: 0 !important; box-sizing: border-box !important; padding: 0 10px; overflow: visible !important;
}
.glp-ui .lm-TabBar-tabLabel, .glp-ui .p-TabBar-tabLabel { flex: 0 0 auto !important; overflow: visible !important; white-space: nowrap; }
@TABWIDTHS@
.glp-ui .widget-tab-contents, .glp-ui .jupyter-widget-tab-contents { padding: 0 !important; }
.glp-ui .glp-disclosure.widget-toggle-button, .glp-ui .glp-disclosure.widget-toggle-button.mod-active,
.glp-ui .glp-disclosure.jupyter-button, .glp-ui .glp-disclosure.jupyter-button.mod-active {
  background: transparent; box-shadow: none; border: none; padding: 0; text-align: left; justify-content: flex-start;
}
"""


@dataclass(frozen=True, slots=True)
class ColumnGrid:
    """Place panel elements on whole columns, so every row starts and ends its elements on the same lines."""

    track_px: int = TRACK_PX
    gap_px: int = GAP_PX

    def span(self, width_px: float) -> int:
        """Return the fewest whole columns whose width holds ``width_px``."""
        if not math.isfinite(width_px) or width_px < 0:
            message = f"A grid width must be finite and non-negative, got {width_px!r}."
            raise ValueError(message)
        return max(1, math.ceil((width_px + self.gap_px) / (self.track_px + self.gap_px)))

    def width(self, span: int) -> int:
        """Return the pixel width of an element spanning ``span`` columns, inner gaps included."""
        if isinstance(span, bool) or not isinstance(span, int) or span < 1:
            message = f"A grid span must be a positive integer, got {span!r}."
            raise ValueError(message)
        return span * self.track_px + (span - 1) * self.gap_px

    def offset(self, columns: int) -> int:
        """Return the pixel offset of the start of column ``columns`` (zero-based) from the left line."""
        if isinstance(columns, bool) or not isinstance(columns, int) or columns < 0:
            message = f"A grid column offset needs a non-negative integer, got {columns!r}."
            raise ValueError(message)
        return columns * (self.track_px + self.gap_px)

    def snap(self, width_px: float) -> int:
        """Return ``width_px`` rounded up to the next column boundary."""
        return self.width(self.span(width_px))


GRID: Final = ColumnGrid()


def text_ch(text: str) -> float:
    """Return a generous width estimate of ``text`` in ch: upper case counts 1.25, m, w, M, W and @ count 1.5."""
    return sum(_WIDE_CH if char in _WIDE_CHARACTERS else _UPPER_CH if char.isupper() else 1.0 for char in text)


def text_px(text: str, *, allowance_ch: float = 0.0) -> int:
    """Return the estimated pixel width of ``text`` plus an allowance in ch."""
    return math.ceil((text_ch(text) + allowance_ch) * CH_PX)


def tab_class(width_ch: int) -> str:
    """Return the class that gives every header of a tab set ``width_ch`` ch."""
    if isinstance(width_ch, bool) or not isinstance(width_ch, int) or not 1 <= width_ch <= TAB_WIDTH_CH_MAX:
        message = f"A tab width must be an integer between 1 and {TAB_WIDTH_CH_MAX} ch, got {width_ch!r}."
        raise ValueError(message)
    return f"glp-tabw-{width_ch}"


def stylesheet() -> str:
    """Return the scoped stylesheet with every constant substituted, as one ``<style>`` element."""
    tab_widths = "\n".join(
        f".glp-ui.{tab_class(width)} .lm-TabBar-tab, .glp-ui.{tab_class(width)} .p-TabBar-tab {{ width: {width}ch !important; }}"
        for width in range(1, TAB_WIDTH_CH_MAX + 1)
    )
    replacements = {
        "@TABWIDTHS@": tab_widths,
        "@LINE@": LINE_PX,
        "@ROWGAP@": ROW_GAP_PX,
        "@GAP@": GAP_PX,
        "@LABELGAP@": LABEL_GAP_PX,
        "@RADIOGAP@": RADIO_GAP_PX,
        "@STEPGAP@": STEP_GAP_PX,
        "@STATUS@": STATUS_PX,
        "@PLOTGAP@": PLOT_GAP_PX,
        "@PITCH@": ROW_PITCH_PX,
        "@TABH@": TAB_HEIGHT_PX,
    }
    text = _STYLESHEET
    for token, value in replacements.items():
        text = text.replace(token, str(value))
    return f"<style>{text}</style>"


def stylesheet_widget() -> widgets.HTML:
    """Return a zero-height HTML widget that carries the stylesheet into the notebook output."""
    carrier = widgets.HTML(stylesheet(), layout=widgets.Layout(height="0px", margin="0", padding="0", overflow="hidden"))
    carrier.add_class(ROOT_CLASS)
    return carrier
