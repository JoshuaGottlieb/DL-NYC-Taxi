from typing import Optional

from matplotlib.axes import Axes

def snake_to_title(label: str) -> str:
    """
    Convert a snake_case string into Title Case with spaces, but do not
    capitalize the first letter after an apostrophe.

    Args:
        label (str): Input string in snake_case format.

    Returns:
        str: Converted string in Title Case with spaces, preserving apostrophes.
    """
    # Replace underscores with spaces
    label = label.replace("_", " ")

    # Capitalize each word, but leave letters after apostrophes lowercase
    # Ex: cohen's_omega -> Cohen's Omega
    def cap_word(word: str) -> str:
        if "'" in word:
            parts = word.split("'")
            # Capitalize first part, keep the rest as-is
            return parts[0].capitalize() + "'" + "'".join(parts[1:])
        else:
            return word.capitalize()

    return " ".join(cap_word(w) for w in label.split())

def snake_to_title_axes(ax: Axes, x: bool = True, y: bool = True) -> None:
    """
    Convert the x-axis and/or y-axis labels of a matplotlib Axes object 
    from snake_case to title case.

    Args:
        ax (matplotlib.axes.Axes): The axes object whose labels will be updated.
        x (bool, optional): If True, convert the x-axis label. Defaults to True.
        y (bool, optional): If True, convert the y-axis label. Defaults to True.
    """
    # Convert x-axis label to title case if requested
    if x:
        ax.set_xlabel(snake_to_title(ax.get_xlabel()))

    # Convert y-axis label to title case if requested
    if y:
        ax.set_ylabel(snake_to_title(ax.get_ylabel()))

    return

def snake_to_title_ticks(ax: Axes, x: bool = True, y: bool = True,
                         rotation_x: int = 0, rotation_y: int = 0) -> None:
    """
    Convert x- and y-axis tick labels to Title Case with optional tick rotation formatting.

    Args:
        ax (matplotlib.axes.Axes): Axes object to modify.
        x (bool, optional): Apply conversion to x-axis tick labels. Defaults to True.
        y (bool, optional): Apply conversion to y-axis tick labels. Defaults to True.
        rotation_x (int, optional): Apply rotation to x-axis tick labels. Defaults to 0 (no rotation).
        rotation_y (int, optional): Apply rotation to y-axis tick labels. Defaults to 0 (no rotation).
    """    
    # Convert x-axis labels
    if x:
        ax.tick_params(axis = 'x', labelrotation = rotation_x)
        ax.set_xticks(ax.get_xticks()) # Needed to suppress Matplotlib warnings
        ax.set_xticklabels([snake_to_title(lbl.get_text()) for lbl in ax.get_xticklabels()])

    # Convert y-axis labels
    if y:
        ax.tick_params(axis = 'y', labelrotation = rotation_y)
        ax.set_yticks(ax.get_yticks()) # Needed to suppress Matplotlib warnings
        ax.set_yticklabels([snake_to_title(lbl.get_text()) for lbl in ax.get_yticklabels()])

    return

def generate_title(x: str, y: Optional[str] = None, hue: Optional[str] = None, stat: Optional[str] = None) -> str:
    """
    Generate a standardized, title-cased plot title based on inputs.

    Args:
        x (str): Name of x-axis variable.
        y (str, optional): Name of y-axis variable. Defaults to None.
        hue (str, optional): Name of hue/grouping variable. Defaults to None.
        stat (str, optional): Statistic name to include (e.g., 'count'). Defaults to None.

    Returns:
        str: Formatted plot title.
    """
    parts = []

    if stat:
        parts.append(snake_to_title(stat))

    if y:
        parts.append(f"{snake_to_title(y)} by {snake_to_title(x)}")
    else:
        parts.append(f"{snake_to_title(x)}")

    if hue:
        parts.append(f"Grouped by {snake_to_title(hue)}")

    return " ".join(parts)