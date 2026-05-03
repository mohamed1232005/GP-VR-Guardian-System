def new_session() -> dict:
    return {
        "state": "INIT",
        "floor": None,
        "boundary_pts": [],
    }