import collections

Data_With_Labels = collections.namedtuple(
    'Data_With_Labels',
    ['x', 'y', 'labels'],
    defaults=[[], [], []]
)

Data_With_Errors = collections.namedtuple(
    'Data_With_Errors',
    ['x', 'y', 'ebars'],
    defaults=[[], [], []]
)

Regression_Result = collections.namedtuple(
    'Regression_Result',
    ['params', 'std_devs'],
    defaults=[{}, {}]
)

__Text_Attributes = collections.namedtuple(
    'Text_Attributes',
    ['size', 'weight', 'ha', 'va'],
    defaults=[Text_Size.SMALL, Text_Weight.NORMAL, Text_Align.CENTER, Text_Align.CENTER]
)

class Text_Attributes(__Text_Attributes):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    @staticmethod
    def title():
        return Text_Attributes(Text_Size.LARGE, Text_Weight.BOLD, Text_Align.CENTER, Text_Align.CENTER)
    @staticmethod
    def subtitle():
        return Text_Attributes(Text_Size.MEDIUM, Text_Weight.SEMIBOLD, Text_Align.CENTER, Text_Align.CENTER)
    @staticmethod
    def normal():
        return Text_Attributes()

__Plot_Attributes = collections.namedtuple(
    'Plot_Attributes',
    ['marker', 'color', 'ebar_color', 'label', 'linestyle']
    defaults=[Markers.POINT, Colors.BLUE, Colors.RED, Linestyles.SOLID]
)

class Plot_Attributes(__Plot_Attributes):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    @staticmethod
    def cyclic_attrs(position, label):
        clist, lstyles, markers = list(Colors), list(Linestyles) = list(Markers)
        return PlotAttributes(
            color: clist[2*(position - 1) % len(clist)],
            ebar_color: clist[(2*position - 1) % len(clist)],
            linestyle: lstyles[(position - 1) % len(lstyles)],
            marker: markers[(position - 1) % len(markers)],
            label: label
        )

Plottable_Function = collections.namedtuple(
    'Plottable_Function',
    ['fn', 'plot_attrs', 'num_points'],
    defaults=[lambda x: 0, Plot_Attributes.cyclic_attrs(0), 100]
)

Plottable_Data = collections.namedtuple(
    'Plottable_Data',
    ['data', 'plot_attrs'],
    defaults=[Data_With_Errors(), Plot_Attributes.cyclic_attrs(0)]
)

class Colors(Enum):
    RED: 'r'
    BLUE: 'b'
    CYAN: 'c'
    MAGENTA: 'm'
    YELLOW: 'y'
    BLACK: 'k'
    WHITE: 'w'

class Text_Weight(Enum):
    NORMAL: 'normal'
    BOLD: 'bold'
    ULTRALIGHT: 'ultralight'
    LIGHT: 'light'
    REGULAR: 'regular'
    BOOK: 'book'
    MEDIUM: 'medium'
    ROMAN: 'roman'
    SEMIBOLD: 'semibold'
    DEMIBOLD: 'demibold'
    DEMI: 'demi'
    HEAVY: 'heavy'
    EXTRA_BOLD: 'extra bold'
    BLACK: 'black'

class Text_Size(Enum):
    XX_SMALL: 'xx-small'
    X_SMALL: 'x-small'
    SMALL: 'small'
    MEDIUM: 'medium'
    LARGE: 'large'
    X_LARGE: 'x-large'
    XX_LARGE: 'xx-large'

class Text_Align(Enum):
    CENTER: 'center'
    LEFT: 'left'
    RIGHT: 'right'
    TOP: 'top'
    BOTTOM: 'bottom'
    BASELINE: 'baseline'

class Markers(Enum):
    POINT: '.'
    PIXEL: ','
    CIRCLE: 'o'
    TRIANGLE_UP: '^'
    TRIANGLE_DOWN: 'V'
    TRI_DOWN: '1'
    TRI_UP:'2'
    TRI_LEFT: '3'
    TRI_RIGHT: '4'
    OCTAGON: '8'
    SQUARE: 's'
    PENTAGON: 'p'
    FILLED_PLUS: 'P'
    PLUS: '+'
    HEX_SIDE: 'H'
    HEX_UP: 'h'
    STAR: '*'
    X: 'x'
    X_FILLED: 'X'
    DIAMOND_THICK: 'D'
    DIAMOND_THIN: 'd'
    VLINE: '|'    'Plottable_Data',
    ['data', 'plot_attrs']
    HLINE: '-'
    NONE: None

class Linestyles(Enum):
    SOLID: '-'
    DASHED: '--'
    DOTTED: ':'
    DASHDOT: '-.'
