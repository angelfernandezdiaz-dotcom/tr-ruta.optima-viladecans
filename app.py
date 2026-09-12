"""
FASE 3 - Aplicació web interactiva amb Streamlit
Simulador de Dijkstra sobre la xarxa viària de Viladecans.
Visualitza l'exploració progressiva de l'algorisme (front d'ona),
la ruta òptima, i aporta validació científica (benchmark vs NetworkX)
i exportació de resultats per a la memòria del TR.
"""

import heapq
import json
import math
import os
import re
import time
from datetime import datetime

import osmnx as ox
ox.settings.bidirectional_network_types = ["walk"]
import networkx as nx
import pydeck as pdk
import streamlit as st

import folium
from streamlit_folium import st_folium

# =============================================================================
# 1. CONFIGURACIÓ DE LA PÀGINA
# =============================================================================

st.set_page_config(layout="wide", page_title="Simulador d'algorismes de ruta òptima a Viladecans")


# =============================================================================
# 2. CÀRREGA DEL GRAF AMB CACHÉ
# =============================================================================

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
LUGAR_DEFECTO = "Viladecans, Barcelona, Spain"
VELOCIDAD_DEFECTO_KMH = 30
VELOCITAT_COTXE_KMH = 50
VELOCITAT_CAMINANT_KMH = 4.5
PENALIZACION_METROS = 15
UMBRAL_CONEXIONES = 6
CRITERIO_DEFECTO = "Distància més curta (Metres)"
PESO_UNICO = "length"


@st.cache_data(show_spinner=False)
def carregar_graf_des_de_fitxer(network_type):
    # 1. Buscar a la mateixa carpeta que app.py o a la carpeta cache/
    arrel_projecte = os.path.dirname(os.path.abspath(__file__))
    candidats = [
        os.path.join(arrel_projecte, f"Viladecans_Spain_{network_type}.graphml"),
        os.path.join(CACHE_DIR, f"Viladecans_Spain_{network_type}.graphml"),
    ]

    # 2. Si el fitxer existeix, carregar-lo directament sense cap connexió externa
    for ruta in candidats:
        if os.path.exists(ruta):
            return ox.load_graphml(ruta)

    # 3. Sense fitxer local: aturar-se, NO hi ha cap descàrrega externa
    st.error(f"Falta el fitxer Viladecans_Spain_{network_type}.graphml a GitHub.")
    st.stop()


def preparar_graf(network_type, penalitzar_cruces):
    """
    Carrega el graf segons el mode (drive/walk), en fa una còpia
    de treball per no contaminar la memòria cau, força la
    bidireccionalitat completa si és caminant, i l'enriqueix
    amb dades de velocitat i temps.

    - Mode "Cotxe" (drive): graf dirigit (MultiDiGraph) respectant
      estrictament el sentit únic i restriccions de les vies.
    - Mode "Caminant" (walk): converteix el graf en bidireccional
      pur per garantir la circulació en ambdues direccions per
      qualsevol carrer.
    """
    G_base = carregar_graf_des_de_fitxer(network_type)
    G = G_base.copy()

    if network_type == 'walk':
        G = G.to_undirected().to_directed()

    G = enriquecer_grafo(
        G, network_type=network_type, penalizar_cruces=penalitzar_cruces
    )
    return G


# =============================================================================
# 3. ENRIQUIMENT DEL GRAF (VELOCITAT I TEMPS)
# =============================================================================

def _limpiar_maxspeed(maxspeed_raw):
    """
    Estandarditza l'atribut maxspeed d'OSMnx (str, llista o None).
    Retorna la velocitat en km/h o None si no es pot determinar.
    """
    if maxspeed_raw is None:
        return None

    if isinstance(maxspeed_raw, list):
        if not maxspeed_raw:
            return None
        maxspeed_raw = maxspeed_raw[0]

    if isinstance(maxspeed_raw, str):
        match = re.search(r"\d+", maxspeed_raw.strip().lower())
        if match:
            return int(match.group(0))
        return None

    if isinstance(maxspeed_raw, (int, float)):
        return int(maxspeed_raw)

    return None


def _grado_conexiones(G):
    """Diccionari node -> nombre d'arestes incidents (grau)."""
    return dict(G.degree())


@st.cache_data(show_spinner=False)
def enriquecer_grafo(
    _G,
    network_type: str = "drive",
    penalizar_cruces=False,
    velocidad_default=VELOCIDAD_DEFECTO_KMH,
):
    """
    Afegeix l'atribut 'tiempo_seg' a cada aresta:
        tiempo_seg = length / (maxspeed_kmh * 1000 / 3600)
    Si maxspeed és nul o llista, s'usa la velocitat per defecte (30 km/h).

    Si penalizar_cruces=True, se sumen 15 metres a les arestes que
    connecten nodes amb un alt nombre de connexions (simulant semàfors).

    El paràmetre network_type (drive/walk) participa en la clau de
    memòria cau de Streamlit: en canviar de mode, la funció es torna
    a avaluar amb el graf corresponent. El graf _G no es calcula el
    seu hash (guionet baix) per evitar errors UnhashableParamError.
    """
    G2 = _G.copy()
    grados = _grado_conexiones(G2)

    for u, v, key, data in G2.edges(keys=True, data=True):
        length = data.get("length", 0)
        if not length or length <= 0:
            length = 1.0

        maxspeed_kmh = _limpiar_maxspeed(data.get("maxspeed"))
        if maxspeed_kmh is None:
            maxspeed_kmh = velocidad_default

        velocidad_ms = maxspeed_kmh * 1000 / 3600
        tiempo_seg = length / velocidad_ms

        if penalizar_cruces:
            if grados.get(u, 0) >= UMBRAL_CONEXIONES or grados.get(v, 0) >= UMBRAL_CONEXIONES:
                length += PENALIZACION_METROS

        G2[u][v][key]["length"] = round(length, 3)
        G2[u][v][key]["tiempo_seg"] = round(tiempo_seg, 3)

    return G2

# =============================================================================
# 4. DIJKSTRA: CUA DE PRIORITATS AMB g(n)
# =============================================================================

def executar_dijkstra(G, origen, desti):
    """
    Algorisme de Dijkstra sobre el cost real acumulat g(n).

    Utilitza una cua de prioritats (Min-Heap amb heapq) on la prioritat és
    g(n), el cost acumulat des de l'origen. Inicialitza la distància de
    l'origen a 0 i la resta a infinit. En cada iteració extreu el node amb
    menor g(n), registra la ramificació/exploració per a l'animació visual
    i avalua els seus veïns adjacents.

    g(n) s'obté sempre de l'atribut 'length' (distància en metres) de les
    arestes: la distància és l'únic criteri d'optimització.

    L'algorisme s'atura quan el node destí s'extreu de la cua: en aquest
    moment la seva distància és la mínima garantida (teoria de grafs).

    Retorna una tupla (ruta, cost, nodes_explorats, historial):
      - ruta: llista de nodes des de l'origen fins al destí
      - cost: g(n) acumulat total fins al destí
      - nodes_explorats: nombre de nodes extrets de la cua
      - historial: llista de dicts {'u', 'v', 'distancia'} amb les arestes
        registrades per a l'animació visual
    """
    distancias = {nodo: float("inf") for nodo in G.nodes()}
    distancias[origen] = 0.0

    padres = {nodo: None for nodo in G.nodes()}

    pq = [(0, origen)]
    historial = []
    nodes_explorats = 0

    while pq:
        dist_actual, u = heapq.heappop(pq)
        nodes_explorats += 1

        # 1. PARADA ESTRICTA: si el node extret és el destí, hem acabat.
        if u == desti:
            break

        # 2. Ignorar entrades obsoletes de la cua (distància no millorada).
        if dist_actual > distancias.get(u, float('inf')):
            continue

        # 3. REGISTRE VISUAL: només quan el node s'assenta (g(n) fixada).
        if u in padres and u != origen:
            pare = padres[u]
            if isinstance(G, (nx.MultiDiGraph, nx.MultiGraph)):
                pes_visual = G[pare][u][0].get('length', 1.0)
            else:
                pes_visual = G[pare][u].get('length', 1.0)

            historial.append({
                'u': pare,
                'v': u,
                'distancia': round(pes_visual, 2)
            })

        # 4. AVALUAR VEÏNS: relaxació amb el cost real g(n) = length (metres).
        for v, datos_arista in G[u].items():
            if isinstance(G, (nx.MultiDiGraph, nx.MultiGraph)):
                pes_arista = datos_arista[0].get('length', 1.0)
            else:
                pes_arista = datos_arista.get('length', 1.0)

            nova_distancia = dist_actual + pes_arista

            if nova_distancia < distancias.get(v, float('inf')):
                distancias[v] = nova_distancia
                padres[v] = u
                heapq.heappush(pq, (nova_distancia, v))

    if distancias[desti] == float("inf"):
        return [], float("inf"), nodes_explorats, historial

    ruta = []
    nodo = desti
    while nodo is not None:
        ruta.insert(0, nodo)
        nodo = padres[nodo]

    return ruta, distancias[desti], nodes_explorats, historial


# =============================================================================
# 4b. A*: CUA DE PRIORITATS AMB f(n) = g(n) + h(n)
# =============================================================================

def heuristica_euclidiana(node_actual, node_desti, G):
    # Extreure coordenades (latitud i longitud en graus)
    lat1, lon1 = G.nodes[node_actual]["y"], G.nodes[node_actual]["x"]
    lat2, lon2 = G.nodes[node_desti]["y"], G.nodes[node_desti]["x"]

    # Factor de conversió de graus a metres (aprox. 111.000 metres per grau de latitud)
    lat_mitjana = math.radians((lat1 + lat2) / 2.0)
    dx = (lon2 - lon1) * math.cos(lat_mitjana) * 111000.0
    dy = (lat2 - lat1) * 111000.0

    # Distància en línia recta en METRES (admissible i consistent amb g(n))
    return math.sqrt(dx * dx + dy * dy)


def executar_astar(G, origen, desti):
    """
    Algorisme A* amb la mateixa estructura que Dijkstra però amb prioritat
    f(n) = g(n) + h(n), on h(n) és la distància euclidiana en línia recta
    fins al destí. El càlcul de g(n) és idèntic al de Dijkstra i s'obté
    sempre de l'atribut 'length' (distància en metres).

    Retorna la mateixa estructura de dades que executar_dijkstra:
    (ruta, cost, nodes_explorats, historial).
    """
    distancias = {nodo: float("inf") for nodo in G.nodes()}
    distancias[origen] = 0.0

    padres = {nodo: None for nodo in G.nodes()}

    # La cua guarda (f_score, g_score, nodo)
    h_origen = heuristica_euclidiana(origen, desti, G)
    pq = [(h_origen, 0, origen)]
    historial = []
    nodes_explorats = 0

    while pq:
        f_actual, dist_actual, u = heapq.heappop(pq)
        nodes_explorats += 1

        # 1. PARADA ESTRICTA: si el node extret és el destí, hem acabat.
        if u == desti:
            break

        # 2. Ignorar entrades obsoletes de la cua (g(n) no millorada).
        if dist_actual > distancias.get(u, float('inf')):
            continue

        # 3. REGISTRE VISUAL: només quan el node s'assenta.
        if u in padres and u != origen:
            pare = padres[u]
            if isinstance(G, (nx.MultiDiGraph, nx.MultiGraph)):
                pes_visual = G[pare][u][0].get('length', 1.0)
            else:
                pes_visual = G[pare][u].get('length', 1.0)

            historial.append({
                'u': pare,
                'v': u,
                'distancia': round(pes_visual, 2)
            })

        # 4. AVALUAR VEÏNS: g(n) = length (metres), exactament igual que a Dijkstra.
        for v, datos_arista in G[u].items():
            if isinstance(G, (nx.MultiDiGraph, nx.MultiGraph)):
                pes_arista = datos_arista[0].get('length', 1.0)
            else:
                pes_arista = datos_arista.get('length', 1.0)

            nova_distancia = dist_actual + pes_arista

            if nova_distancia < distancias.get(v, float('inf')):
                distancias[v] = nova_distancia
                padres[v] = u

                # A* ordena per f(n) = g(n) + h(n)
                h_score = heuristica_euclidiana(v, desti, G)
                f_score = nova_distancia + h_score

                heapq.heappush(pq, (f_score, nova_distancia, v))

    if distancias[desti] == float("inf"):
        return [], float("inf"), nodes_explorats, historial

    ruta = []
    nodo = desti
    while nodo is not None:
        ruta.insert(0, nodo)
        nodo = padres[nodo]

    return ruta, distancias[desti], nodes_explorats, historial


# =============================================================================
# 5. VALIDACIÓ CIENTÍFICA (BENCHMARK VS NETWORKX)
# =============================================================================

def validar_con_networkx(G, nodo_origen, nodo_destino, algorisme):
    """
    Executa la funció autòctona de NetworkX corresponent a l'algorisme
    seleccionat i mesura el temps en ms. El pes (weight) és sempre 'length'.

    - Dijkstra: nx.dijkstra_path i nx.dijkstra_path_length
    - A*: nx.astar_path i nx.astar_path_length

    Retorna: (ruta, coste_total_segun_peso, tiempo_ms)
    """
    if algorisme == "Dijkstra":
        calcula_ruta = nx.dijkstra_path
        calcula_coste = nx.dijkstra_path_length
    else:
        calcula_ruta = nx.astar_path
        calcula_coste = nx.astar_path_length

    inicio = time.perf_counter()
    ruta = calcula_ruta(G, source=nodo_origen, target=nodo_destino, weight="length")
    coste = calcula_coste(G, source=nodo_origen, target=nodo_destino, weight="length")
    fin = time.perf_counter()

    return ruta, coste, (fin - inicio) * 1000


def metricas_ruta(G, ruta, velocitat_kmh):
    """
    Calcula la distància (m) real d'una ruta i el temps estimat (min)
    segons la velocitat mitjana del mode de transport (50 km/h cotxe,
    4.5 km/h caminant).
    """
    longitud = 0.0

    for u, v in zip(ruta, ruta[1:]):
        aristas_uv = G[u][v]
        if hasattr(aristas_uv, "items"):
            for _key, data in aristas_uv.items():
                if isinstance(data, dict):
                    longitud += data.get("length", 0) or 0
                    break
        elif isinstance(aristas_uv, dict):
            longitud += aristas_uv.get("length", 0) or 0

    tempo_seg = longitud / (velocitat_kmh * 1000 / 3600)
    return longitud, tempo_seg / 60.0


def calcular_coincidencia(ruta_propia, ruta_nx):
    """
    Calcula el percentatge d'arestes compartides entre totes dues rutes.
    Si són idèntiques, retorna el 100%.
    """
    if not ruta_propia or not ruta_nx:
        return 0.0

    aristas_propias = set(zip(ruta_propia, ruta_propia[1:]))
    aristas_nx = set(zip(ruta_nx, ruta_nx[1:]))

    comunes = len(aristas_propias & aristas_nx)
    total = max(len(aristas_propias), len(aristas_nx))

    return 100.0 * comunes / total if total else 0.0


# =============================================================================
# 6. UTILITATS PER A LA VISUALITZACIÓ (GEOMETRIA D'ARESTES / PYDECK)
# =============================================================================

def _geometria_arista_lonlat(G, u, v):
    """
    Retorna la llista de parells [lon, lat] que descriuen la geometria de
    l'aresta (u, v). Usa l'atribut 'geometry' d'OSMnx si existeix; si no,
    usa les coordenades directes dels nodes u i v.
    Aquest ordre [lon, lat] és el requerit per pdk.PathLayer.
    """
    datos = None
    aristas_uv = G[u][v]
    if hasattr(aristas_uv, "items"):
        for _key, data in aristas_uv.items():
            if isinstance(data, dict):
                datos = data
                break
    elif isinstance(aristas_uv, dict):
        datos = aristas_uv

    geometria = datos.get("geometry") if datos else None

    if geometria is not None:
        return [[lon, lat] for lon, lat in geometria.coords]

    return [
        [G.nodes[u]["x"], G.nodes[u]["y"]],
        [G.nodes[v]["x"], G.nodes[v]["y"]],
    ]


def preparar_rutas_pydeck(G, historial, color, width):
    """
    Converteix l'historial d'arestes (u, v) en l'estructura estricta
    exigida per pdk.PathLayer. Cada tram és un diccionari amb:

      {
        "path": [[lon1, lat1], [lon2, lat2], ...],
        "color": [r, g, b, a],
        "width": n
      }
    """
    rutas = []
    for entrada in historial:
        try:
            if isinstance(entrada, dict):
                u = entrada["u"]
                v = entrada["v"]
            else:
                u, v = entrada
            path = _geometria_arista_lonlat(G, u, v)
        except (KeyError, TypeError):
            continue
        rutas.append({"path": path, "color": color, "width": width})
    return rutas


def construir_deck(G, datos_ramificaciones, datos_ruta, nodo_origen, nodo_destino):
    """
    Munta el pdk.Deck complet amb:
      - Capa base: tota la xarxa viària en gris tenue (no seleccionable).
      - Capa front d'ona: ramificacions explorades en taronja/vermell semi-transparent.
      - Capa ruta: ruta òptima en blau intens amb més gruix.
      - Marcadors d'origen i destí.
    """
    # Capa base: tota la xarxa viària en gris clar de fons
    base_aristas = list(G.edges())
    datos_base = preparar_rutas_pydeck(G, base_aristas, [160, 160, 160, 50], 2)
    capa_base = pdk.Layer(
        "PathLayer",
        data=datos_base,
        get_path="path",
        get_color="color",
        get_width="width",
        width_min_pixels=2,
        pickable=False,
    )

    # Capa d'exploració: front d'ona en taronja/vermell semi-transparent
    capa_exploracion = pdk.Layer(
        "PathLayer",
        data=datos_ramificaciones,
        get_path="path",
        get_color="color",
        get_width="width",
        width_min_pixels=2,
        pickable=True,
    )

    # Capa de la ruta òptima: blau intens i més gruixuda
    capa_ruta = pdk.Layer(
        "PathLayer",
        data=datos_ruta,
        get_path="path",
        get_color="color",
        get_width="width",
        width_min_pixels=2,
        pickable=True,
    )

    # Marcadors d'origen i destí
    lat_o = G.nodes[nodo_origen]["y"]
    lon_o = G.nodes[nodo_origen]["x"]
    lat_d = G.nodes[nodo_destino]["y"]
    lon_d = G.nodes[nodo_destino]["x"]

    capa_origen = pdk.Layer(
        "ScatterplotLayer",
        data=[{"position": [lon_o, lat_o]}],
        get_position="position",
        get_radius=10,
        radius_min_pixels=4,
        radius_max_pixels=15,
        get_fill_color=[0, 255, 0, 255],
        pickable=False,
    )
    capa_destino = pdk.Layer(
        "ScatterplotLayer",
        data=[{"position": [lon_d, lat_d]}],
        get_position="position",
        get_radius=10,
        radius_min_pixels=4,
        radius_max_pixels=15,
        get_fill_color=[255, 0, 0, 255],
        pickable=False,
    )

    lat_centro = (lat_o + lat_d) / 2
    lon_centro = (lon_o + lon_d) / 2

    vista = pdk.ViewState(
        latitude=lat_centro,
        longitude=lon_centro,
        zoom=14,
        pitch=0,
    )

    return pdk.Deck(
        map_style=None,
        initial_view_state=vista,
        layers=[
            capa_base,
            capa_exploracion,
            capa_ruta,
            capa_origen,
            capa_destino,
        ],
    )


# =============================================================================
# 6b. MAPA INTERACTIU FOLIUM PER A LA SELECCIÓ D'ORIGEN I DESTÍ
# =============================================================================

def obtenir_graf_per_seleccio(tipus_xarxa, penalitzar_cruces):
    """
    Retorna el graf de treball del mode seleccionat, reutilitzant-lo a la
    memòria de sessió mentre no canviï el mode de transport ni la
    penalització per semàfors. S'utilitza per capturar els clics del mapa
    (ox.nearest_nodes) sense tornar a carregar el graf a cada rerun.
    """
    clau = (tipus_xarxa, penalitzar_cruces)
    if st.session_state.get("clau_graf_seleccio") != clau:
        with st.spinner("Carregant la xarxa viària per al mapa interactiu..."):
            st.session_state["G_seleccio"] = preparar_graf(tipus_xarxa, penalitzar_cruces)
        st.session_state["clau_graf_seleccio"] = clau
    return st.session_state["G_seleccio"]


def construir_mapa_seleccio(lat_origen, lon_origen, lat_destino, lon_destino):
    """
    Munta el mapa interactiu de Folium amb dos marcadors dinàmics:
    - Verd (play): Origen
    - Vermell (flag): Destí

    La vista es centra en el punt mitjà entre Origen i Destí amb zoom 14.
    El component només retorna last_clicked (els pan/zoom no provoquen
    reruns), així que la posició del mapa es manté si el marcador no canvia.
    """
    ubicacio = [(lat_origen + lat_destino) / 2, (lon_origen + lon_destino) / 2]

    mapa_sel = folium.Map(location=ubicacio, zoom_start=14, tiles="OpenStreetMap")

    folium.Marker(
        [lat_origen, lon_origen],
        popup="Origen",
        tooltip="Origen",
        icon=folium.Icon(color="green", icon="play", prefix="fa"),
    ).add_to(mapa_sel)

    folium.Marker(
        [lat_destino, lon_destino],
        popup="Destí",
        tooltip="Destí",
        icon=folium.Icon(color="red", icon="flag", prefix="fa"),
    ).add_to(mapa_sel)

    return mapa_sel


# =============================================================================
# 7. EXPORTACIÓ DE RESULTATS (JSON PER AL TR)
# =============================================================================

def generar_resumen_json(resultado):
    """
    Genera el JSON tècnic descarregable amb el resum de la prova.
    Serveix com a evidència directa per annexar a la memòria del TR.
    """
    resumen = {
        "descripcion": "Simulador Dijkstra - Viladecans (Fase 3)",
        "timestamp": resultado["timestamp"],
        "lugar": LUGAR_DEFECTO,
        "coordenadas_origen": {
            "lat": resultado["lat_origen"],
            "lon": resultado["lon_origen"],
        },
        "coordenadas_destino": {
            "lat": resultado["lat_destino"],
            "lon": resultado["lon_destino"],
        },
        "criterio_optimizacion": resultado["criterio"],
        "peso_usado": resultado["peso"],
        "penalizacion_semaforos": resultado["penalizar"],
        "penalizacion_aplicada_m": PENALIZACION_METROS if resultado["penalizar"] else 0,
        "velocidad_por_defecto_kmh": VELOCIDAD_DEFECTO_KMH,
        "resultados": {
            "distancia_total_m": round(resultado["distancia_total"], 2),
            "tiempo_total_min": round(resultado["tiempo_ruta_min"], 2),
            "total_exploraciones": resultado["total_exploraciones"],
            "total_ramificaciones_registradas": len(resultado["ramificaciones_datos"]),
            "nodos_en_ruta": len(resultado["ruta_optima"]),
            "nodos_origen_id": resultado["nodo_origen"],
            "nodos_destino_id": resultado["nodo_destino"],
        },
        "benchmark": {
            "tiempo_dijkstra_propio_ms": round(resultado["t_propio_ms"], 3),
            "tiempo_networkx_ms": round(resultado["t_nx_ms"], 3),
            "coincidencia_ruta_pct": round(resultado["pct_coincidencia"], 2),
            "coste_networkx": round(resultado["coste_nx"], 3),
        },
    }
    return json.dumps(resumen, ensure_ascii=False, indent=2)


def generar_gpx(G, ruta_nodes):
    """
    Genera el codi XML d'un fitxer GPX (v1.1) amb la ruta calculada.
    Cada node de la ruta es converteix en un punt trkpt del segment <trkseg>.
    """
    gpx = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<gpx version="1.1" creator="TR-Viladecans" xmlns="http://www.topografix.com/GPX/1/1">',
        "  <trk>",
        "    <name>Ruta Optima Viladecans</name>",
        "    <trkseg>",
    ]
    for node in ruta_nodes:
        lat = G.nodes[node]["y"]
        lon = G.nodes[node]["x"]
        gpx.append(f'      <trkpt lat="{lat}" lon="{lon}"></trkpt>')
    gpx.append("    </trkseg>")
    gpx.append("  </trk>")
    gpx.append("</gpx>")
    return "\n".join(gpx)


# =============================================================================
# 8. BARRA LATERAL: ENTRADES DE L'USUARI
# =============================================================================

# ---------------------------------------------------------------------------
# ESTAT CANÒNIC DE LES COORDENADES I DEL CONTROL DE CLICS (st.session_state)
# ---------------------------------------------------------------------------
if "origen_coords" not in st.session_state:
    st.session_state["origen_coords"] = (41.315, 2.011)  # Coordenades per defecte
if "desti_coords" not in st.session_state:
    st.session_state["desti_coords"] = (41.320, 2.018)
if "ultim_clic_processat" not in st.session_state:
    st.session_state["ultim_clic_processat"] = None
if "seleccio_actual" not in st.session_state:
    st.session_state["seleccio_actual"] = "Origen"
if 'simular' not in st.session_state:
    st.session_state.simular = False


# ---------------------------------------------------------------------------
# PROCESSAR EL CLIC PENDENT DEL MAPA A L'INICI DEL SCRIPT
# (abans de dibuixar cap widget, per evitar StreamlitAPIException per
#  reassignació de l'estat d'un widget ja instanciat)
# ---------------------------------------------------------------------------
clic_pendent = st.session_state.get("clic_pendent")
if clic_pendent is not None:
    st.session_state["clic_pendent"] = None

    lat_p, lon_p = clic_pendent

    # Valors persistits en una execució anterior (els widgets ja s'han
    # dibuixat abans), per tant són segurs de llegir aquí.
    tipus_xarxa_top = (
        "drive"
        if st.session_state.get("mode_selector", "Cotxe") == "Cotxe"
        else "walk"
    )
    graf_seleccio = obtenir_graf_per_seleccio(
        tipus_xarxa_top, st.session_state.get("penalitzar_cruces", False)
    )

    nodo_seleccionat = ox.nearest_nodes(graf_seleccio, X=lon_p, Y=lat_p)
    lat_node = graf_seleccio.nodes[nodo_seleccionat]["y"]
    lon_node = graf_seleccio.nodes[nodo_seleccionat]["x"]

    # Actualitzar la tupla canònica i les claus dels widgets (abans de crear-los)
    if st.session_state.seleccio_actual == "Origen":
        st.session_state["origen_coords"] = (lat_node, lon_node)
        st.session_state["widget_lat_origen"] = lat_node
        st.session_state["widget_lon_origen"] = lon_node
    else:
        st.session_state["desti_coords"] = (lat_node, lon_node)
        st.session_state["widget_lat_destino"] = lat_node
        st.session_state["widget_lon_destino"] = lon_node

    st.session_state["ultim_clic_processat"] = (
        st.session_state.seleccio_actual, round(lat_p, 7), round(lon_p, 7)
    )
    st.rerun()  # Els widgets agafaran els nous valors al re-instantciar-se


def intercanviar_coordenades():
    # S'executa ABANS de dibuixar els widgets
    st.session_state.origen_coords, st.session_state.desti_coords = (
        st.session_state.desti_coords, st.session_state.origen_coords
    )
    st.session_state.widget_lat_origen, st.session_state.widget_lat_destino = (
        st.session_state.widget_lat_destino, st.session_state.widget_lat_origen
    )
    st.session_state.widget_lon_origen, st.session_state.widget_lon_destino = (
        st.session_state.widget_lon_destino, st.session_state.widget_lon_origen
    )


def canvi_mode():
    # Quan l'usuari canvia de cotxe a caminant (o viceversa), torna a
    # calcular automàticament amb el nou mode sense recarregar la pàgina.
    st.session_state['simular'] = True
    st.session_state['resultados'] = None
    if 'G' in st.session_state:
        del st.session_state['G']


with st.sidebar:
    st.sidebar.title("Simulador d'algorismes a Viladecans")
    st.markdown(
        """
        **Com funcionen aquests algorismes?**

        *   **Dijkstra:** Explora la xarxa de carrers en totes direccions alhora, creant un "front d'ona" circular. Garanteix trobar la ruta més curta, però a costa d'explorar molts carrers innecessaris que s'allunyen del destí.
        *   **A* (A-Estrella):** És la versió "intel·ligent" de Dijkstra. Utilitza una estimació matemàtica (heurística) basada en la distància en línia recta fins al destí. D'aquesta manera, prioritza explorar només els carrers que l'acosten a la meta, garantint trobar també la ruta òptima però explorant molts menys nodes.
        """
    )
    st.markdown(
        """
        **Treball de Recerca** - Optimització de rutes amb els
        algorismes de Dijkstra i A* sobre la xarxa viària de Viladecans.

        Tria el criteri d'optimització, ajusta la velocitat de
        reproducció i prem **Inicia la Simulació**.
        """
    )

    st.subheader("Coordenades d'origen")
    lat_origen = st.number_input(
        "Latitud d'origen",
        format="%.6f",
        value=st.session_state.origen_coords[0],
        key="widget_lat_origen",
    )
    lon_origen = st.number_input(
        "Longitud d'origen",
        format="%.6f",
        value=st.session_state.origen_coords[1],
        key="widget_lon_origen",
    )
    st.session_state["origen_coords"] = (lat_origen, lon_origen)

    st.button(
        "🔄 Intercanviar origen i destí",
        on_click=intercanviar_coordenades,
        use_container_width=True,
    )

    st.subheader("Coordenades de destí")
    lat_destino = st.number_input(
        "Latitud de destí",
        format="%.6f",
        value=st.session_state.desti_coords[0],
        key="widget_lat_destino",
    )
    lon_destino = st.number_input(
        "Longitud de destí",
        format="%.6f",
        value=st.session_state.desti_coords[1],
        key="widget_lon_destino",
    )
    st.session_state["desti_coords"] = (lat_destino, lon_destino)

    st.radio(
        "Assignar clic al mapa a:",
        options=["Origen", "Destí"],
        key="seleccio_actual",
        horizontal=True,
        help="Els clics sobre el mapa interactiu actualitzaran la coordenada seleccionada.",
    )

    st.selectbox(
        "Mode de transport",
        ["Cotxe", "Caminant"],
        key="mode_selector",
        on_change=canvi_mode,
    )
    tipus_xarxa = "drive" if st.session_state["mode_selector"] == "Cotxe" else "walk"

    algorisme_triat = st.selectbox("Tria l'algorisme", ["Dijkstra", "A* (A-Star)"])

    st.caption("Optimització per distància (metres).")

    penalizar_cruces = st.toggle(
        "Simular penalització per semàfors/encreuaments (+15 m per intersecció)",
        value=False,
        key="penalitzar_cruces",
    )

    mida_lot = st.slider(
        "Carrers per fotograma (Mida del lot)",
        min_value=1,
        max_value=200,
        value=15,
        step=1,
        help="Més alt = més ràpid",
    )

    pausa_animacio = st.slider(
        "Pausa entre fotogrames (segons)",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        help="Més alt = més lent",
    )

    boto_iniciar = st.sidebar.button("▶️ Iniciar simulació", type="primary")

    if boto_iniciar:
        st.session_state.simular = True
        st.session_state["resultados"] = None

    st.divider()

    # Botó de descàrrega del resum per al TR (només si hi ha resultats)
    resultados_guardados = st.session_state.get("resultados")
    if resultados_guardados and resultados_guardados.get("ruta_optima"):
        resumen_json = generar_resumen_json(resultados_guardados)
        nombre_archivo = (
            "resumen_dijkstra_"
            + resultados_guardados["timestamp"].replace(":", "-").replace(" ", "_")
            + ".json"
        )
        st.download_button(
            label="Descarrega el resum (JSON)",
            data=resumen_json,
            file_name=nombre_archivo,
            mime="application/json",
            use_container_width=True,
        )

        gpx = generar_gpx(
            resultados_guardados["G"], resultados_guardados["ruta_optima"]
        )
        nombre_gpx = (
            "ruta_optima_viladecans_"
            + resultados_guardados["timestamp"].replace(":", "-").replace(" ", "_")
            + ".gpx"
        )
        st.download_button(
            label="Descarrega la ruta (GPX)",
            data=gpx,
            file_name=nombre_gpx,
            mime="application/gpx+xml",
            use_container_width=True,
        )


# =============================================================================
# 9. ZONA PRINCIPAL: CÀRREGA, CÀLCUL, ANIMACIÓ I VALIDACIÓ
# =============================================================================

st.title("Simulador d'algorismes de ruta òptima a Viladecans")

# =============================================================================
# 9a. MAPA INTERACTIU PER SELECCIONAR ORIGEN I DESTINACIÓ (CLIC DIRECTE)
# =============================================================================

st.markdown("## 🗺️ Selecció interactiva d'origen i destí")
st.caption(
    f"📌 Fes clic sobre el mapa per assignar automàticament el punt "
    f"(desplaçat al node més proper de la xarxa viària) a **{st.session_state.seleccio_actual}**. "
    "Els marcadors verd (Origen) i vermell (Destí) es mouen automàticament."
)

# ---------------------------------------------------------------
# El clic es processa a l'inici del script (tuples origen_coords/desti_coords).
# Aquí només es dibuixa el mapa Folium amb les coordenades actuals.
# ---------------------------------------------------------------
mapa_seleccio = construir_mapa_seleccio(
    st.session_state.origen_coords[0],
    st.session_state.origen_coords[1],
    st.session_state.desti_coords[0],
    st.session_state.desti_coords[1],
)

# RENDERITZAR EL MAPA I CAPTURAR EL NOU CLIC (es desa per al proper rerun)
dades_mapa = st_folium(
    mapa_seleccio,
    width=700,
    height=500,
    returned_objects=["last_clicked"],
    key="mapa_dinamic",
)

clic = dades_mapa.get("last_clicked")
if clic and clic.get("lat") is not None and clic.get("lng") is not None:
    lat_clic = float(clic["lat"])
    lon_clic = float(clic["lng"])
    clau_clic = (
        st.session_state.seleccio_actual, round(lat_clic, 7), round(lon_clic, 7),
    )

    if st.session_state.get("ultim_clic_processat") != clau_clic:
        st.session_state["clic_pendent"] = (lat_clic, lon_clic)
        st.rerun()

st.divider()

if not st.session_state.simular:
    st.info(
        """
        Configura les coordenades i els paràmetres a la barra lateral i prem
        **▶️ Iniciar simulació** per executar la simulació.
        """
    )
else:
    firma = (lat_origen, lon_origen, lat_destino, lon_destino, penalizar_cruces, tipus_xarxa, algorisme_triat)
    resultado = st.session_state.get("resultados")

    if resultado is None:
        # Primer càlcul o botó premut: calcular la simulació
        with st.spinner("Carregant el graf i calculant rutes..."):
            G = preparar_graf(tipus_xarxa, penalizar_cruces)
            st.session_state['G'] = G

            nodo_origen = ox.nearest_nodes(G, X=lon_origen, Y=lat_origen)
            nodo_destino = ox.nearest_nodes(G, X=lon_destino, Y=lat_destino)

            inicio = time.perf_counter()
            if algorisme_triat == "A* (A-Star)":
                ruta_optima, distancia_total, nusos_explorats, historial = executar_astar(G, nodo_origen, nodo_destino)
            else:
                ruta_optima, distancia_total, nusos_explorats, historial = executar_dijkstra(G, nodo_origen, nodo_destino)
            fin = time.perf_counter()
            t_propio_ms = (fin - inicio) * 1000

            velocitat_mode = (
                VELOCITAT_COTXE_KMH if tipus_xarxa == "drive" else VELOCITAT_CAMINANT_KMH
            )

            try:
                ruta_nx, coste_nx, t_nx_ms = validar_con_networkx(
                    G, nodo_origen, nodo_destino, algorisme_triat
                )
            except nx.NetworkXNoPath:
                ruta_nx, coste_nx, t_nx_ms = [], float("inf"), 0.0

            distancia_ruta, tiempo_ruta_min = metricas_ruta(G, ruta_optima, velocitat_mode)
            distancia_nx, tiempo_nx_min = metricas_ruta(G, ruta_nx, velocitat_mode)

            pct_coincidencia = calcular_coincidencia(ruta_optima, ruta_nx)

        if not ruta_optima:
            resultado = {
                "firma": firma,
                "ruta_optima": [],
                "criterio": CRITERIO_DEFECTO,
                "algorisme": algorisme_triat,
                "peso": PESO_UNICO,
                "penalizar": penalizar_cruces,
                "lat_origen": lat_origen,
                "lon_origen": lon_origen,
                "lat_destino": lat_destino,
                "lon_destino": lon_destino,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            st.session_state["resultados"] = resultado
        else:
            # Preparar dades de visualització (estructura estricta Pydeck)
            ramificaciones_datos = preparar_rutas_pydeck(
                G, historial, color=[255, 69, 0, 150], width=3
            )

            aristas_ruta = [
                (ruta_optima[i], ruta_optima[i + 1])
                for i in range(len(ruta_optima) - 1)
            ]
            ruta_datos = preparar_rutas_pydeck(
                G, aristas_ruta, color=[30, 144, 255, 255], width=8
            )

            deck_final = construir_deck(
                G, ramificaciones_datos, ruta_datos, nodo_origen, nodo_destino
            )

            resultado = {
                "firma": firma,
                "G": G,
                "nodo_origen": nodo_origen,
                "nodo_destino": nodo_destino,
                "ruta_optima": ruta_optima,
                "historial": historial,
                "distancia_total": distancia_total,
                "total_exploraciones": len(historial),
                "nusos_explorats": nusos_explorats,
                "ramificaciones_datos": ramificaciones_datos,
                "ruta_datos": ruta_datos,
                "deck_final": deck_final,
                "t_propio_ms": t_propio_ms,
                "ruta_nx": ruta_nx,
                "coste_nx": coste_nx,
                "t_nx_ms": t_nx_ms,
                "distancia_ruta": distancia_ruta,
                "tiempo_ruta_min": tiempo_ruta_min,
                "distancia_nx": distancia_nx,
                "tiempo_nx_min": tiempo_nx_min,
                "pct_coincidencia": pct_coincidencia,
                "criterio": CRITERIO_DEFECTO,
                "algorisme": algorisme_triat,
                "peso": PESO_UNICO,
                "penalizar": penalizar_cruces,
                "lat_origen": lat_origen,
                "lon_origen": lon_origen,
                "lat_destino": lat_destino,
                "lon_destino": lon_destino,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "reproducir_animacion": True,
            }
            st.session_state["resultados"] = resultado
    elif resultado.get("firma") != firma:
        # L'usuari ha modificat algun paràmetre: posem la simulació en pausa
        st.session_state.simular = False
        st.info(
            """
            S'han detectat canvis en els paràmetres o les coordenades. Prem
            **▶️ Iniciar simulació** per tornar a executar la simulació.
            """
        )
        resultado = None

    if resultado is not None:
        if not resultado["ruta_optima"]:
            st.error("No s'ha trobat cap ruta entre les coordenades indicades.")
        else:
            # -------------------------------------------------------
            # ANIMACIÓ PAS A PAS: creixement del front d'ona
            # -------------------------------------------------------
            mapa_placeholder = st.empty()
            ramificaciones_datos = resultado["ramificaciones_datos"]
            G_deck = resultado["G"]

            if resultado.get("reproducir_animacion"):
                n_ramificaciones = len(ramificaciones_datos)
                mida_lot_actiu = mida_lot if n_ramificaciones else 1

                barra_progreso = st.progress(0.0)

                for i in range(0, n_ramificaciones, mida_lot_actiu):
                    datos_lote = ramificaciones_datos[:i + mida_lot_actiu]
                    deck = construir_deck(
                        G_deck,
                        datos_lote,
                        [],
                        resultado["nodo_origen"],
                        resultado["nodo_destino"],
                    )
                    mapa_placeholder.pydeck_chart(deck)
                    barra_progreso.progress(min((i + mida_lot_actiu) / n_ramificaciones, 1.0))
                    time.sleep(pausa_animacio)

                barra_progreso.progress(1.0)
                mapa_placeholder.pydeck_chart(resultado["deck_final"])
                resultado["reproducir_animacion"] = False
            else:
                # Rerun sense canvis de paràmetres: mostrar l'estat final
                mapa_placeholder.pydeck_chart(resultado["deck_final"])

            st.success("Simulació completada. Ruta òptima ressaltada en blau intens.")

            # -------------------------------------------------------
            # MÈTRIQUES EN PANTALLA
            # -------------------------------------------------------
            distancia = resultado["distancia_total"]

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Distància de la ruta òptima",
                f"{resultado['distancia_ruta'] / 1000:.2f} km"
                if resultado['distancia_ruta'] >= 1000
                else f"{resultado['distancia_ruta']:.0f} m",
            )
            col2.metric(
                "Temps estimat",
                f"{resultado['tiempo_ruta_min']:.2f} min",
            )
            col3.metric(
                "Ramificacions avaluades",
                f"{len(ramificaciones_datos)}",
            )
            col4.metric(
                "Nodes a la ruta òptima",
                f"{len(resultado['ruta_optima'])}",
            )

            nom_mode = "cotxe (50 km/h)" if tipus_xarxa == "drive" else "caminant (4.5 km/h)"
            st.caption(f"Temps estimat amb velocitat mitjana de {nom_mode}.")
            st.caption(f"Criteri: {resultado['criterio']} | Pes: length (cost total = {distancia:.1f} m)")

            # -------------------------------------------------------
            # VALIDACIÓ CIENTÍFICA I BENCHMARK
            # -------------------------------------------------------
            st.markdown(f"### Benchmark / Validació: {algorisme_triat} vs NetworkX")

            etiqueta_algorisme = f"Algorisme propi ({algorisme_triat})"

            comparativa = [
                {
                    "Mètrica": "Distància total (m)",
                    etiqueta_algorisme: f"{resultado['distancia_ruta']:.1f}",
                    "NetworkX": f"{resultado['distancia_nx']:.1f}",
                    "Diferència": f"{resultado['distancia_ruta'] - resultado['distancia_nx']:+.1f}",
                },
                {
                    "Mètrica": "Coincidència de ruta",
                    etiqueta_algorisme: "100% (referència)",
                    "NetworkX": f"{resultado['pct_coincidencia']:.2f}%",
                    "Diferència": (
                        "Rutes idèntiques" if resultado['pct_coincidencia'] >= 100.0 else "Rutes diferents"
                    ),
                },
            ]

            # Renderitzat en Markdown pur per evitar la dependència de pyarrow
            columnes_taula = list(comparativa[0].keys())
            linies_md = ["| " + " | ".join(columnes_taula) + " |"]
            linies_md.append("| " + " | ".join(["---"] * len(columnes_taula)) + " |")
            for fila_comparativa in comparativa:
                linies_md.append(
                    "| " + " | ".join(str(fila_comparativa[col]) for col in columnes_taula) + " |"
                )
            st.markdown("\n".join(linies_md))
            st.caption("Taula comparativa: algorisme manual vs funció autòctona de NetworkX.")

            if resultado["pct_coincidencia"] >= 100.0:
                st.success(
                    "Verificació d'exactitud superada: la ruta del teu algorisme és "
                    "100% idèntica a la ruta nativa de NetworkX."
                )
            else:
                st.warning(
                    f"Ambdues rutes coincideixen en un {resultado['pct_coincidencia']:.2f}% "
                    "de les arestes (verificació d'exactitud parcial, possible empat en cost)."
                )
