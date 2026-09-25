"""Published optimal tour lengths for TSPLIB95 instances.

Values are the optima listed on the TSPLIB95 site (``STSP.html`` / ``ATSP.html``)
for the symmetric and asymmetric collections, in each instance's own integer
metric. Only instances with proven optima are listed; instances not in this
table are loaded with ``optimum=None`` and scored against a reference solver.

Checked against the TSPLIB solutions list and, for 77 instances with n <= 1000
plus all 19 ATSP instances, against Concorde. The one disagreement is d657:
Concorde proves 48913 under the EUC_2D rounding it and ``tspbench`` both
implement, one unit above the published 48912; the published value is kept.
"""

STSP_OPTIMA = {
    "a280": 2579, "ali535": 202339, "att48": 10628, "att532": 27686,
    "bayg29": 1610, "bays29": 2020, "berlin52": 7542, "bier127": 118282,
    "brazil58": 25395, "brd14051": 469385, "brg180": 1950, "burma14": 3323,
    "ch130": 6110, "ch150": 6528, "d198": 15780, "d493": 35002, "d657": 48912,
    "d1291": 50801, "d1655": 62128, "d2103": 80450, "d15112": 1573084,
    "d18512": 645238, "dantzig42": 699, "dsj1000": 18660188, "eil51": 426, "eil76": 538,
    "eil101": 629, "fl417": 11861, "fl1400": 20127, "fl1577": 22249,
    "fl3795": 28772, "fnl4461": 182566, "fri26": 937, "gil262": 2378,
    "gr17": 2085, "gr21": 2707, "gr24": 1272, "gr48": 5046, "gr96": 55209,
    "gr120": 6942, "gr137": 69853, "gr202": 40160, "gr229": 134602,
    "gr431": 171414, "gr666": 294358, "hk48": 11461, "kroA100": 21282,
    "kroB100": 22141, "kroC100": 20749, "kroD100": 21294, "kroE100": 22068,
    "kroA150": 26524, "kroB150": 26130, "kroA200": 29368, "kroB200": 29437,
    "lin105": 14379, "lin318": 42029, "nrw1379": 56638, "p654": 34643,
    "pa561": 2763, "pcb442": 50778, "pcb1173": 56892, "pcb3038": 137694,
    "pla7397": 23260728, "pla33810": 66048945, "pla85900": 142382641,
    "pr76": 108159, "pr107": 44303, "pr124": 59030, "pr136": 96772,
    "pr144": 58537, "pr152": 73682, "pr226": 80369, "pr264": 49135,
    "pr299": 48191, "pr439": 107217, "pr1002": 259045, "pr2392": 378032,
    "rat99": 1211, "rat195": 2323, "rat575": 6773, "rat783": 8806,
    "rd100": 7910, "rd400": 15281, "rl1304": 252948, "rl1323": 270199,
    "rl1889": 316536, "rl5915": 565530, "rl5934": 556045, "rl11849": 923288,
    "si175": 21407, "si535": 48450, "si1032": 92650, "st70": 675,
    "swiss42": 1273, "ts225": 126643, "tsp225": 3916, "u159": 42080,
    "u574": 36905, "u724": 41910, "u1060": 224094, "u1432": 152970,
    "u1817": 57201, "u2152": 64253, "u2319": 234256, "ulysses16": 6859,
    "ulysses22": 7013, "usa13509": 19982859, "vm1084": 239297,
    "vm1748": 336556,
}

ATSP_OPTIMA = {
    "br17": 39, "ft53": 6905, "ft70": 38673, "ftv33": 1286, "ftv35": 1473,
    "ftv38": 1530, "ftv44": 1613, "ftv47": 1776, "ftv55": 1608, "ftv64": 1839,
    "ftv70": 1950, "ftv170": 2755, "kro124p": 36230, "p43": 5620,
    "rbg323": 1326, "rbg358": 1163, "rbg403": 2465, "rbg443": 2720,
    "ry48p": 14422,
}


def optimum(name: str):
    return STSP_OPTIMA.get(name, ATSP_OPTIMA.get(name))
