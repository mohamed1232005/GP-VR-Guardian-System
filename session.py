"""Per-client session state. Intentionally plain — no logic here."""

def new_session():
    return {
        "state"         : "INIT",       # INIT | PLACING_POINTS | GUARDIAN_READY
        "floor"         : None,         # dict from FLOOR_CONFIRM payload
        "boundary_pts"  : [],           # list of [x, y, z] world points
    }