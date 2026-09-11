"""
FASE 3 - Aplicacion web interactiva con Streamlit
Simulador de Dijkstra sobre la red viaria de Viladecans.
Visualiza la exploracion progresiva del algoritmo (frente de onda),
la ruta optima, y aporta validacion cientifica (benchmark vs NetworkX)
y exportacion de resultados para la memoria del TR.
"""

import heapq
import json
import os
import re
import time
from datetime import datetime

import osmnx as ox
import networkx as nx
import pydeck as pdk
import streamlit as st

# =============================================================================
# 1. CONFIGURACION DE LA PAGINA
# =============================================================================

st.set_page_config(layout="wide", page_title="Simulador Dijkstra - Viladecans")


# =============================================================================
# 2. CARGA DEL GRAFO CON CACHEO
# =============================================================================

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
LUGAR_DEFECTO = "Viladecans, Barcelona, Spain"
VELOCIDAD_DEFECTO_KMH = 30
PENALIZACION_INT = 15
UMBRAL_CONEXIONES = 6


@st.cache_data(show_spinner=False)
def descargar_grafo(lugar=LUGAR_DEFECTO, network_type="drive"):
    """
    Descarga el grafo de la red viaria de Viladecans con cacheo local.
    Si el grafo ya esta en CACHE_DIR, lo carga directamente.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    nombre_archivo = lugar.replace(", ", "_").replace(" ", "_")
    ruta_cache = os.path.join(CACHE_DIR, f"{nombre_archivo}.graphml")

    if os.path.exists(ruta_cache):
        G = ox.load_graphml(ruta_cache)
    else:
        G = ox.graph_from_place(lugar, network_type=network_type)
        ox.save_graphml(G, filepath=ruta_cache)

    return G


# =============================================================================
# 3. ENRIQUECIMIENTO DEL GRAFO (VELOCIDAD Y TIEMPOS)
# =============================================================================

def _limpiar_maxspeed(maxspeed_raw):
    """
    Estandariza el atributo maxspeed de OSMnx (str, lista o None).
    Retorna la velocidad en km/h o None si no se puede determinar.
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
    """Diccionario nodo -> numero de aristas incidentes (grado)."""
    return dict(G.degree())


@st.cache_data(show_spinner=False)
def enriquecer_grafo(_G, penalizar_cruces=False, velocidad_default=VELOCIDAD_DEFECTO_KMH):
    """
    Anade el atributo 'tiempo_seg' a cada arista:
        tiempo_seg = length / (maxspeed_kmh * 1000 / 3600)
    Si maxspeed es nulo o lista, se usa la velocidad por defecto (30 km/h).

    Si penalizar_cruces=True, se suman 15 segundos en las aristas que
    conectan nodos con un alto numero de conexiones (simulando semaforos).
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
                tiempo_seg += PENALIZACION_INT

        G2[u][v][key]["tiempo_seg"] = round(tiempo_seg, 3)

    return G2


# =============================================================================
# 4. DIJKSTRA CON REGISTRO TOTAL DE RAMIFICACIONES
# =============================================================================

def dijkstra_con_ramificaciones(G, nodo_origen, nodo_destino, peso="length"):
    """
    Dijkstra con registro de ramificaciones y parada teorica inmediata.

    El algoritmo se detiene en cuanto el nodo destino es extraido de la
    cola de prioridad (teoria de grafos: es su distancia minima garantizada).

    Retorna un diccionario con:
      - ruta_optima: lista de nodos desde origen hasta destino
      - historial_ramificaciones: lista de diccionarios {'u', 'v', 'distancia'}
        con las aristas realmente exploradas (relajadas) por el algoritmo
      - distancia_total: float con la distancia acumulada
      - total_exploraciones: entero con el numero total de aristas evaluadas
    """
    distancias = {nodo: float("inf") for nodo in G.nodes()}
    distancias[nodo_origen] = 0.0

    padres = {nodo: None for nodo in G.nodes()}

    pq = [(0, nodo_origen)]
    historial_ramificaciones = []
    total_exploraciones = 0

    while pq:
        dist_actual, u = heapq.heappop(pq)
       
        # 1. CONDICIÓN DE PARADA TEÓRICA: Si el nodo extraído es el destino, terminamos.
        if u == nodo_destino:
            break

        # 2. Ignorar si ya hemos procesado este nodo con una distancia menor
        if dist_actual > distancias.get(u, float("inf")):
            continue

        # 3. Exploración de vecinos
        for v, datos_arista in G[u].items():
            total_exploraciones += 1

            # Calcular 'peso_arista' a partir de los datos de la arista
            pesos = []
            if hasattr(datos_arista, "items"):
                for _key, data in datos_arista.items():
                    if isinstance(data, dict):
                        w = data.get(peso, data.get("length", 1))
                        if isinstance(w, (int, float)):
                            pesos.append(w)
            elif isinstance(datos_arista, dict):
                w = datos_arista.get(peso, datos_arista.get("length", 1))
                if isinstance(w, (int, float)):
                    pesos.append(w)

            peso_arista = min(pesos) if pesos else 1.0
            nueva_distancia = dist_actual + peso_arista
        historial_ramificaciones.append({
            'u': u,
            'v': v,
            'distancia': round(peso_arista, 1) # (o la variable que usaras para la distancia)
        })   
            if nueva_distancia < distancias.get(v, float("inf")):
                distancias[v] = nueva_distancia
                padres[v] = u
                heapq.heappush(pq, (nueva_distancia, v))
         # ---> ANTES ESTABA AQUÍ (Vuelve a ponerlo exactamente aquí) <---
                     
                

    if distancias[nodo_destino] == float("inf"):
        return {
            "ruta_optima": [],
            "historial_ramificaciones": historial_ramificaciones,
            "distancia_total": float("inf"),
            "total_exploraciones": total_exploraciones,
        }

    ruta = []
    nodo = nodo_destino
    while nodo is not None:
        ruta.insert(0, nodo)
        nodo = padres[nodo]

    return {
        "ruta_optima": ruta,
        "historial_ramificaciones": historial_ramificaciones,
        "distancia_total": distancias[nodo_destino],
        "total_exploraciones": total_exploraciones,
    }


# =============================================================================
# 5. VALIDACION CIENTIFICA (BENCHMARK VS NETWORKX)
# =============================================================================

def validar_con_networkx(G, nodo_origen, nodo_destino, peso):
    """
    Ejecuta el algoritmo nativo de NetworkX y mide el tiempo en ms.
    Retorna: (ruta, coste_total_segun_peso, tiempo_ms)
    """
    inicio = time.perf_counter()
    ruta = nx.shortest_path(G, source=nodo_origen, target=nodo_destino, weight=peso)
    coste = nx.shortest_path_length(G, source=nodo_origen, target=nodo_destino, weight=peso)
    fin = time.perf_counter()

    return ruta, coste, (fin - inicio) * 1000


def metricas_ruta(G, ruta):
    """
    Calcula la distancia (m) real y el tiempo estimado (min) de una ruta,
    independientemente del peso usado por Dijkstra.
    """
    longitud = 0.0
    tiempo = 0.0

    for u, v in zip(ruta, ruta[1:]):
        aristas_uv = G[u][v]
        if hasattr(aristas_uv, "items"):
            for _key, data in aristas_uv.items():
                if isinstance(data, dict):
                    longitud += data.get("length", 0) or 0
                    tiempo += data.get("tiempo_seg", 0) or 0
                    break
        elif isinstance(aristas_uv, dict):
            longitud += aristas_uv.get("length", 0) or 0
            tiempo += aristas_uv.get("tiempo_seg", 0) or 0

    return longitud, tiempo / 60.0


def calcular_coincidencia(ruta_propia, ruta_nx):
    """
    Calcula el porcentaje de aristas compartidas entre ambas rutas.
    Si son identicas, devuelve 100%.
    """
    if not ruta_propia or not ruta_nx:
        return 0.0

    aristas_propias = set(zip(ruta_propia, ruta_propia[1:]))
    aristas_nx = set(zip(ruta_nx, ruta_nx[1:]))

    comunes = len(aristas_propias & aristas_nx)
    total = max(len(aristas_propias), len(aristas_nx))

    return 100.0 * comunes / total if total else 0.0


# =============================================================================
# 6. UTILIDADES PARA LA VISUALIZACION (GEOMETRIA DE ARISTAS / PYDECK)
# =============================================================================

def _geometria_arista_lonlat(G, u, v):
    """
    Devuelve la lista de pares [lon, lat] que describen la geometria de la
    arista (u, v). Usa el atributo 'geometry' de OSMnx si existe; si no,
    usa las coordenadas directas de los nodos u y v.
    Este orden [lon, lat] es el requerido por pdk.PathLayer.
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
    Convierte el historial de aristas (u, v) en la estructura estricta
    exigida por pdk.PathLayer. Cada tramo es un diccionario con:

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
    Ensambla el pdk.Deck completo con:
      - Capa base: toda la red viaria en gris tenue (no seleccionable).
      - Capa frente de onda: ramificaciones en azul semi-transparente.
      - Capa ruta: ruta optima en rojo brillante con mayor grosor.
      - Marcadores de origen y destino.
    """
    # Capa base: toda la red viaria en gris claro de fondo
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

    # Capa de exploracion: frente de onda en azul semi-transparente
    capa_exploracion = pdk.Layer(
        "PathLayer",
        data=datos_ramificaciones,
        get_path="path",
        get_color="color",
        get_width="width",
        width_min_pixels=2,
        pickable=True,
    )

    # Capa de la ruta optima: rojo brillante y mas gruesa
    capa_ruta = pdk.Layer(
        "PathLayer",
        data=datos_ruta,
        get_path="path",
        get_color="color",
        get_width="width",
        width_min_pixels=2,
        pickable=True,
    )

    # Marcadores de origen y destino
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
# 7. EXPORTACION DE RESULTADOS (JSON PARA EL TR)
# =============================================================================

def generar_resumen_json(resultado):
    """
    Genera el JSON tecnico descargable con el resumen de la prueba.
    Sirve como evidencia directa para anexar en la memoria del TR.
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
        "penalizacion_aplicada_seg": PENALIZACION_INT if resultado["penalizar"] else 0,
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


# =============================================================================
# 8. BARRA LATERAL: ENTRADAS DEL USUARIO
# =============================================================================

with st.sidebar:
    st.title("Simulador Dijkstra")
    st.markdown(
        """
        **Trabajo de Recerca** - Optimizacion de rutas con el
        algoritmo de Dijkstra sobre la red viaria de Viladecans.

        Elige el criterio de optimizacion, ajusta la velocidad de
        reproduccion y pulsa **Iniciar Simulacion**.
        """
    )

    st.subheader("Coordenadas de Origen")
    lat_origen = st.number_input("Latitud de Origen", value=41.3168, format="%.6f")
    lon_origen = st.number_input("Longitud de Origen", value=2.0163, format="%.6f")

    st.subheader("Coordenadas de Destino")
    lat_destino = st.number_input("Latitud de Destino", value=41.3255, format="%.6f")
    lon_destino = st.number_input("Longitud de Destino", value=2.0005, format="%.6f")

    st.subheader("Criterio de optimización")
    criterio = st.radio(
        "Elige el criterio de la ruta",
        options=["Distancia más corta (Metros)", "Tiempo más rápido (Segundos/Minutos)"],
        index=0,
    )
    peso_elegido = "length" if criterio.startswith("Distancia") else "tiempo_seg"

    penalizar_cruces = st.toggle(
        "Simular penalización por semáforos/cruces (+15s por intersección)",
        value=False,
    )

    velocidad = st.slider(
        "Velocidad de reproducción",
        min_value=1,
        max_value=15,
        value=5,
        step=1,
        format="%dx",
    )

    iniciar = st.button("Iniciar Simulación", type="primary", use_container_width=True)

    if iniciar:
        st.session_state["simulacion_activa"] = True
        st.session_state["resultados"] = None

    st.divider()

    # Boton de descarga del resumen para el TR (solo si hay resultados)
    resultados_guardados = st.session_state.get("resultados")
    if resultados_guardados and resultados_guardados.get("ruta_optima"):
        resumen_json = generar_resumen_json(resultados_guardados)
        nombre_archivo = (
            "resumen_dijkstra_"
            + resultados_guardados["timestamp"].replace(":", "-").replace(" ", "_")
            + ".json"
        )
        st.download_button(
            label="Descargar resumen (JSON)",
            data=resumen_json,
            file_name=nombre_archivo,
            mime="application/json",
            use_container_width=True,
        )


# =============================================================================
# 9. ZONA PRINCIPAL: CARGA, CALCULO, ANIMACION Y VALIDACION
# =============================================================================

st.title("Exploración de Dijkstra en Viladecans")

if not st.session_state.get("simulacion_activa"):
    st.info(
        """
        Configura las coordenadas en la barra lateral y pulsa
        **Iniciar Simulación** para comenzar.
        """
    )
else:
    firma = (lat_origen, lon_origen, lat_destino, lon_destino, peso_elegido, penalizar_cruces)
    resultado = st.session_state.get("resultados")

    # Recalcular solo si el criterio / coords / penalizacion cambian
    if resultado is None or resultado.get("firma") != firma:

        with st.spinner("Enriqueciendo grafo (tiempos) y calculando rutas..."):
            G_base = descargar_grafo()
            G = enriquecer_grafo(G_base, penalizar_cruces=penalizar_cruces)

            nodo_origen = ox.nearest_nodes(G, X=lon_origen, Y=lat_origen)
            nodo_destino = ox.nearest_nodes(G, X=lon_destino, Y=lat_destino)

            inicio = time.perf_counter()
            res_dijkstra = dijkstra_con_ramificaciones(G, nodo_origen, nodo_destino, peso=peso_elegido)
            fin = time.perf_counter()
            t_propio_ms = (fin - inicio) * 1000

            try:
                ruta_nx, coste_nx, t_nx_ms = validar_con_networkx(
                    G, nodo_origen, nodo_destino, peso_elegido
                )
            except nx.NetworkXNoPath:
                ruta_nx, coste_nx, t_nx_ms = [], float("inf"), 0.0

            ruta_optima = res_dijkstra["ruta_optima"]
            distancia_ruta, tiempo_ruta_min = metricas_ruta(G, ruta_optima)
            distancia_nx, tiempo_nx_min = metricas_ruta(G, ruta_nx)

            pct_coincidencia = calcular_coincidencia(ruta_optima, ruta_nx)

        if not ruta_optima:
            st.session_state["resultados"] = {
                "firma": firma,
                "ruta_optima": [],
                "criterio": criterio,
                "peso": peso_elegido,
                "penalizar": penalizar_cruces,
                "lat_origen": lat_origen,
                "lon_origen": lon_origen,
                "lat_destino": lat_destino,
                "lon_destino": lon_destino,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            st.rerun()
        else:
            # Preparar datos de visualizacion (estructura estricta Pydeck)
            historial = res_dijkstra["historial_ramificaciones"]
            ramificaciones_datos = preparar_rutas_pydeck(
                G, historial, color=[30, 144, 255, 180], width=3
            )

            aristas_ruta = [
                (ruta_optima[i], ruta_optima[i + 1])
                for i in range(len(ruta_optima) - 1)
            ]
            ruta_datos = preparar_rutas_pydeck(
                G, aristas_ruta, color=[255, 0, 0, 255], width=8
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
                "distancia_total": res_dijkstra["distancia_total"],
                "total_exploraciones": res_dijkstra["total_exploraciones"],
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
                "criterio": criterio,
                "peso": peso_elegido,
                "penalizar": penalizar_cruces,
                "lat_origen": lat_origen,
                "lon_origen": lon_origen,
                "lat_destino": lat_destino,
                "lon_destino": lon_destino,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "reproducir_animacion": True,
            }
            st.session_state["resultados"] = resultado
            st.rerun()

    resultado = st.session_state.get("resultados")

    if resultado is None:
        st.write("")
    elif not resultado["ruta_optima"]:
        st.error("No se ha encontrado ninguna ruta entre las coordenadas indicadas.")
    else:
        # -------------------------------------------------------
        # ANIMACION PASO A PASO: crecimiento del frente de onda
        # -------------------------------------------------------
        mapa_placeholder = st.empty()
        ramificaciones_datos = resultado["ramificaciones_datos"]
        G_deck = resultado["G"]

        if resultado.get("reproducir_animacion"):
            n_ramificaciones = len(ramificaciones_datos)
            lote = 30 if n_ramificaciones else 1
            pausa = 0.20 / velocidad

            barra_progreso = st.progress(0.0)

            for i in range(0, n_ramificaciones, lote):
                datos_lote = ramificaciones_datos[:i + lote]
                deck = construir_deck(
                    G_deck,
                    datos_lote,
                    [],
                    resultado["nodo_origen"],
                    resultado["nodo_destino"],
                )
                mapa_placeholder.pydeck_chart(deck)
                barra_progreso.progress(min((i + lote) / n_ramificaciones, 1.0))
                time.sleep(pausa)

            barra_progreso.progress(1.0)
            mapa_placeholder.pydeck_chart(resultado["deck_final"])
            resultado["reproducir_animacion"] = False
        else:
            # Rerun sin cambios de parametros: mostrar el estado final
            mapa_placeholder.pydeck_chart(resultado["deck_final"])

        st.success("Simulación completada. Ruta óptima resaltada en rojo.")

        # -------------------------------------------------------
        # METRICAS EN PANTALLA
        # -------------------------------------------------------
        distancia = resultado["distancia_total"]
        peso_criterio = resultado["peso"]

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Distancia de la ruta",
            f"{resultado['distancia_ruta'] / 1000:.2f} km"
            if resultado['distancia_ruta'] >= 10000
            else f"{resultado['distancia_ruta']:.1f} m",
        )
        col2.metric("Tiempo estimado", f"{resultado['tiempo_ruta_min']:.2f} min")
        col3.metric("Ramificaciones evaluadas", f"{len(ramificaciones_datos)}")
        col4.metric("Nodos en la ruta óptima", f"{len(resultado['ruta_optima'])}")

        if peso_criterio == "tiempo_seg":
            st.caption(
                f"Criterio: {resultado['criterio']} | Peso: tiempo_seg "
                f"(coste total = {distancia:.1f} s)"
            )
        else:
            st.caption(
                f"Criterio: {resultado['criterio']} | Peso: length "
                f"(coste total = {distancia:.1f} m)"
            )

        # -------------------------------------------------------
        # VALIDACION CIENTIFICA Y BENCHMARK
        # -------------------------------------------------------
        st.subheader("Validación científica y benchmark")

        unidad_coste = "m" if peso_criterio == "length" else "s"
        etiqueta_coste = f"Coste total ({unidad_coste})"

        comparativa = [
            {
                "Métrica": "Tiempo de ejecución",
                "Algoritmo propio (Dijkstra)": f"{resultado['t_propio_ms']:.3f} ms",
                "NetworkX": f"{resultado['t_nx_ms']:.3f} ms",
                "Diferencia": f"{resultado['t_propio_ms'] - resultado['t_nx_ms']:+.3f} ms",
            },
            {
                "Métrica": etiqueta_coste,
                "Algoritmo propio (Dijkstra)": f"{resultado['distancia_total']:.2f}",
                "NetworkX": f"{resultado['coste_nx']:.2f}",
                "Diferencia": "Coincide" if abs(resultado['distancia_total'] - resultado['coste_nx']) < 1e-6 else "Difiere",
            },
            {
                "Métrica": "Distancia recorrida",
                "Algoritmo propio (Dijkstra)": f"{resultado['distancia_ruta']:.1f} m",
                "NetworkX": f"{resultado['distancia_nx']:.1f} m",
                "Diferencia": f"{resultado['distancia_ruta'] - resultado['distancia_nx']:+.1f} m",
            },
            {
                "Métrica": "Tiempo estimado",
                "Algoritmo propio (Dijkstra)": f"{resultado['tiempo_ruta_min']:.2f} min",
                "NetworkX": f"{resultado['tiempo_nx_min']:.2f} min",
                "Diferencia": f"{resultado['tiempo_ruta_min'] - resultado['tiempo_nx_min']:+.2f} min",
            },
            {
                "Métrica": "Coincidencia de ruta",
                "Algoritmo propio (Dijkstra)": "100% (referencia)",
                "NetworkX": f"{resultado['pct_coincidencia']:.2f}%",
                "Diferencia": (
                    "Rutas idénticas" if resultado['pct_coincidencia'] >= 100.0 else "Rutas distintas"
                ),
            },
        ]

        # Generación manual de tabla Markdown (sin PyArrow)
        if comparativa:
            headers = list(comparativa[0].keys())
            header_row = "| " + " | ".join(headers) + " |"
            separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
            data_rows = ["| " + " | ".join(str(val) for val in row.values()) + " |" for row in comparativa]

            tabla_markdown = "\n".join([header_row, separator_row] + data_rows)
            st.markdown(tabla_markdown)

        if resultado["pct_coincidencia"] >= 100.0:
            st.success(
                "Verificación de exactitud superada: la ruta de tu algoritmo es "
                "100% idéntica a la ruta nativa de NetworkX."
            )
        else:
            st.warning(
                f"Ambas rutas coinciden en un {resultado['pct_coincidencia']:.2f}% "
                "de las aristas (verificacion de exactitud parcial, posible empate en coste)."
            )
