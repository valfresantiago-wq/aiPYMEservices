from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
import openai
import os
import json
import hashlib
import httpx
from typing import List, Optional
import tempfile
from PyPDF2 import PdfReader
import uvicorn
from dotenv import load_dotenv

load_dotenv()

# ============================================
# CONFIGURACIÓN GENERAL
# ============================================
app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

MERCADO_PAGO_ACCESS_TOKEN = os.getenv("MERCADO_PAGO_ACCESS_TOKEN")
PUBLIC_URL = os.getenv("PUBLIC_URL", "https://localhost")

# ============================================
# CONFIGURACIÓN DE BASE DE DATOS (SQLite)
# ============================================
DATABASE_URL = "sqlite:///./stock.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ============================================
# MODELO DE PRODUCTO (Tabla en la DB)
# ============================================
class Producto(Base):
    __tablename__ = "productos"
    
    id = Column(Integer, primary_key=True, index=True)
    nombre = Column(String, index=True, nullable=False)
    descripcion = Column(Text, nullable=True)
    precio = Column(Float, nullable=False)
    costo = Column(Float, nullable=True)
    stock = Column(Integer, nullable=False, default=0)
    categoria = Column(String, index=True, nullable=True)
    etiquetas = Column(String, nullable=True)
    proveedor = Column(String, nullable=True)
    sku = Column(String, unique=True, index=True)
    activo = Column(Boolean, default=True)
    fecha_creacion = Column(DateTime, default=datetime.now)
    fecha_actualizacion = Column(DateTime, default=datetime.now, onupdate=datetime.now)

# Crear las tablas en la base de datos
Base.metadata.create_all(bind=engine)

# ============================================
# BASES DE DATOS EN MEMORIA (para clientes y leads)
# ============================================
clientes_db = {}
pedidos_db = {}
leads_db = {}

# ============================================
# FUNCIONES DE IA (CEREBRO DEL SISTEMA)
# ============================================

def analizar_mensaje(mensaje: str) -> dict:
    """Usa IA para analizar el mensaje y extraer información clave."""
    prompt = f"""
    Eres un asistente de análisis de mensajes para empresas.
    Analiza el siguiente mensaje y extrae la información en formato JSON.
    Mensaje: "{mensaje}"
    
    Responde con un JSON válido con estos campos:
    - intencion: (compra, consulta, reclamo, spam)
    - urgencia: (alta, media, baja)
    - presupuesto_estimado: (bajo, medio, alto)
    - producto_interes: (nombre del producto o servicio que busca)
    - sentimiento: (positivo, neutro, negativo)
    - accion_sugerida: (responder, derivar a humano, enviar cotizacion)
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {
            "intencion": "error",
            "urgencia": "media",
            "presupuesto_estimado": "medio",
            "producto_interes": "",
            "sentimiento": "neutro",
            "accion_sugerida": "responder",
            "error": str(e)
        }

def generar_respuesta(mensaje: str, contexto: dict) -> str:
    """Genera una respuesta personalizada usando IA."""
    prompt = f"""
    Eres un asistente de atención al cliente para una empresa.
    Contexto: {contexto}
    Mensaje del cliente: "{mensaje}"
    Sé amable, directo y útil. Máximo 3 oraciones.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Gracias por tu mensaje. Te atenderemos en breve."

# ============================================
# FUNCIONES DE IA PARA EL CATÁLOGO (STOCK)
# ============================================

def consultar_productos_por_ia(query: str, session: Session) -> List[Producto]:
    """Usa IA para interpretar la pregunta del cliente y buscar productos."""
    prompt = f"""
    Eres un asistente de catálogo. El cliente pregunta: "{query}"
    Responde SOLO con un JSON que contenga:
    - categoria: (string o null)
    - palabras_clave: (lista de strings)
    - precio_max: (float o null)
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content)
        
        query_db = session.query(Producto).filter(Producto.activo == True)
        
        if data.get("categoria"):
            query_db = query_db.filter(Producto.categoria.ilike(f"%{data['categoria']}%"))
        
        if data.get("palabras_clave"):
            for palabra in data["palabras_clave"]:
                query_db = query_db.filter(
                    (Producto.nombre.ilike(f"%{palabra}%")) | 
                    (Producto.descripcion.ilike(f"%{palabra}%")) |
                    (Producto.etiquetas.ilike(f"%{palabra}%"))
                )
        
        if data.get("precio_max"):
            query_db = query_db.filter(Producto.precio <= float(data["precio_max"]))
        
        return query_db.limit(10).all()
    
    except Exception as e:
        return session.query(Producto).filter(
            Producto.nombre.ilike(f"%{query}%") | 
            Producto.descripcion.ilike(f"%{query}%")
        ).limit(5).all()

def generar_respuesta_con_productos(productos: List[Producto], query: str) -> str:
    """Genera una respuesta amigable con los productos encontrados."""
    if not productos:
        return f"Lo siento, no encontré productos que coincidan con '{query}'. ¿Podés ser más específico?"
    
    prompt = f"""
    El cliente preguntó: "{query}"
    Productos encontrados: {json.dumps([{"nombre": p.nombre, "precio": p.precio, "stock": p.stock} for p in productos], indent=2)}
    Genera una respuesta amigable y breve (máximo 3 oraciones) con los precios y stock.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )
        return response.choices[0].message.content
    except:
        respuesta = "Encontré estos productos:\n"
        for p in productos:
            respuesta += f"- {p.nombre}: ${p.precio} ({p.stock} unidades)\n"
        return respuesta

# ============================================
# ENDPOINTS DEL MÓDULO DE STOCK
# ============================================

@app.post("/api/productos")
async def crear_producto(producto_data: dict):
    session = SessionLocal()
    try:
        if not producto_data.get("nombre") or not producto_data.get("precio"):
            raise HTTPException(400, {"error": "Nombre y precio son obligatorios"})
        
        if not producto_data.get("sku"):
            producto_data["sku"] = hashlib.md5(producto_data["nombre"].encode()).hexdigest()[:8].upper()
        
        nuevo_producto = Producto(**producto_data)
        session.add(nuevo_producto)
        session.commit()
        session.refresh(nuevo_producto)
        return {"status": "ok", "producto": {"id": nuevo_producto.id, "nombre": nuevo_producto.nombre}}
    except Exception as e:
        session.rollback()
        raise HTTPException(500, {"error": str(e)})
    finally:
        session.close()

@app.get("/api/productos")
async def listar_productos():
    session = SessionLocal()
    try:
        productos = session.query(Producto).filter(Producto.activo == True).all()
        return {
            "productos": [
                {
                    "id": p.id,
                    "nombre": p.nombre,
                    "precio": p.precio,
                    "stock": p.stock,
                    "categoria": p.categoria,
                }
                for p in productos
            ]
        }
    finally:
        session.close()

@app.get("/api/productos/buscar")
async def buscar_productos(query: str):
    session = SessionLocal()
    try:
        productos = consultar_productos_por_ia(query, session)
        respuesta = generar_respuesta_con_productos(productos, query)
        return {
            "status": "ok",
            "productos": [
                {
                    "nombre": p.nombre,
                    "precio": p.precio,
                    "stock": p.stock,
                    "categoria": p.categoria,
                }
                for p in productos
            ],
            "respuesta": respuesta
        }
    finally:
        session.close()

@app.post("/api/productos/actualizar-stock")
async def actualizar_stock(producto_id: int, cantidad: int):
    session = SessionLocal()
    try:
        producto = session.query(Producto).filter(Producto.id == producto_id).first()
        if not producto:
            raise HTTPException(404, {"error": "Producto no encontrado"})
        
        nuevo_stock = producto.stock + cantidad
        if nuevo_stock < 0:
            raise HTTPException(400, {"error": "Stock insuficiente"})
        
        producto.stock = nuevo_stock
        session.commit()
        return {"status": "ok", "nuevo_stock": nuevo_stock}
    finally:
        session.close()

# ============================================
# WEBHOOK UNIVERSAL (CHAT + CATÁLOGO)
# ============================================

@app.post("/webhook/mensaje")
async def recibir_mensaje(mensaje: dict, background_tasks: BackgroundTasks):
    cliente_id = mensaje.get("cliente_id") or mensaje.get("from")
    texto = mensaje.get("mensaje") or mensaje.get("text", "")
    
    if not cliente_id or not texto:
        raise HTTPException(400, {"error": "Faltan datos"})
    
    # Registrar cliente
    if cliente_id not in clientes_db:
        clientes_db[cliente_id] = {
            "nombre": mensaje.get("nombre", "Cliente"),
            "historial": [],
            "puntaje": 0
        }
    
    clientes_db[cliente_id]["historial"].append({
        "fecha": datetime.now().isoformat(),
        "mensaje": texto,
        "tipo": "recibido"
    })
    
    # Verificar si la pregunta es sobre productos
    palabras_clave = ["precio", "costo", "cuánto", "valor", "tienen", "stock", "disponible", "catalogo", "producto"]
    if any(palabra in texto.lower() for palabra in palabras_clave):
        session = SessionLocal()
        try:
            productos = consultar_productos_por_ia(texto, session)
            respuesta = generar_respuesta_con_productos(productos, texto)
        finally:
            session.close()
    else:
        # Si no es sobre productos, usar el asistente general
        contexto = {"nombre": clientes_db[cliente_id]["nombre"]}
        respuesta = generar_respuesta(texto, contexto)
    
    # Guardar respuesta en historial
    clientes_db[cliente_id]["historial"].append({
        "fecha": datetime.now().isoformat(),
        "mensaje": respuesta,
        "tipo": "enviado"
    })
    
    # Analizar mensaje en background
    background_tasks.add_task(analizar_mensaje_en_background, cliente_id, texto)
    
    return {"status": "ok", "respuesta": respuesta}

async def analizar_mensaje_en_background(cliente_id: str, texto: str):
    analisis = analizar_mensaje(texto)
    if cliente_id in clientes_db:
        clientes_db[cliente_id]["ultimo_analisis"] = analisis
        clientes_db[cliente_id]["puntaje"] = calcular_puntaje(analisis)
        
        if analisis.get("intencion") == "compra" and analisis.get("urgencia") == "alta":
            lead_id = hashlib.md5(f"{cliente_id}{datetime.now().isoformat()}".encode()).hexdigest()
            leads_db[lead_id] = {
                "cliente_id": cliente_id,
                "puntaje": clientes_db[cliente_id]["puntaje"],
                "intencion": analisis.get("intencion"),
                "urgente": True,
                "fecha": datetime.now().isoformat(),
                "producto_interes": analisis.get("producto_interes")
            }
            print(f"🔥 NUEVO LEAD CALIENTE: {cliente_id} - {analisis.get('producto_interes')}")

def calcular_puntaje(analisis: dict) -> int:
    puntaje = 0
    if analisis.get("intencion") == "compra": puntaje += 40
    elif analisis.get("intencion") == "consulta": puntaje += 20
    if analisis.get("urgencia") == "alta": puntaje += 30
    elif analisis.get("urgencia") == "media": puntaje += 15
    if analisis.get("presupuesto_estimado") == "alto": puntaje += 30
    elif analisis.get("presupuesto_estimado") == "medio": puntaje += 15
    if analisis.get("sentimiento") == "positivo": puntaje += 10
    return puntaje

# ============================================
# HEALTH CHECK
# ============================================
@app.get("/health")
async def health():
    session = SessionLocal()
    try:
        total_productos = session.query(Producto).filter(Producto.activo == True).count()
    finally:
        session.close()
    
    return {
        "status": "ok",
        "clientes": len(clientes_db),
        "leads": len(leads_db),
        "productos": total_productos,
        "version": "2.0.0"
    }

# ============================================
# PANEL DE ADMINISTRACIÓN (Frontend)
# ============================================
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sistema de Gestión Inteligente</title>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body { font-family: 'Segoe UI', Tahoma, sans-serif; background: #f0f4f8; padding: 20px; }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { background: white; padding: 30px; border-radius: 24px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); margin-bottom: 30px; }
            .header h1 { font-size: 32px; color: #1a202c; }
            .header p { color: #4a5568; margin-top: 8px; }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
            .card { background: white; padding: 20px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
            .card h2 { font-size: 18px; color: #2d3748; margin-bottom: 12px; }
            .numero { font-size: 36px; font-weight: bold; color: #3182ce; }
            .descripcion { color: #4a5568; font-size: 14px; }
            .lead-caliente { background: #fed7d7; border-left: 4px solid #e53e3e; padding: 10px; border-radius: 8px; margin-bottom: 10px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 Sistema de Gestión Inteligente</h1>
                <p>Panel de administración en tiempo real. Stock, clientes y leads calientes.</p>
            </div>

            <div class="grid">
                <div class="card">
                    <h2>📦 Productos en Stock</h2>
                    <div class="numero" id="totalProductos">0</div>
                    <div class="descripcion">Activos en el catálogo</div>
                </div>
                <div class="card">
                    <h2>👥 Total de Clientes</h2>
                    <div class="numero" id="totalClientes">0</div>
                    <div class="descripcion">Registrados en el sistema</div>
                </div>
                <div class="card">
                    <h2>🔥 Leads Calientes</h2>
                    <div class="numero" id="leadsCalientes">0</div>
                    <div class="descripcion">Listos para cerrar venta</div>
                </div>
            </div>

            <div style="margin-top: 30px;">
                <h2 style="color: #2d3748;">📋 Últimos Leads Calientes</h2>
                <div id="listaLeads" style="margin-top: 12px;">
                    <p style="color: #4a5568;">Cargando datos...</p>
                </div>
            </div>
        </div>

        <script>
            async function actualizarDatos() {
                try {
                    const response = await fetch('/api/dashboard');
                    const data = await response.json();
                    
                    document.getElementById('totalProductos').textContent = data.total_productos || 0;
                    document.getElementById('totalClientes').textContent = data.total_clientes || 0;
                    document.getElementById('leadsCalientes').textContent = data.leads_calientes || 0;
                    
                    if (data.leads && data.leads.length > 0) {
                        const lista = document.getElementById('listaLeads');
                        lista.innerHTML = data.leads.map(lead => `
                            <div class="lead-caliente">
                                <strong>${lead.cliente_id}</strong> - ${lead.producto_interes || 'Producto no especificado'}
                                <span style="float: right; background: #e53e3e; color: white; padding: 2px 12px; border-radius: 12px; font-size: 12px;">
                                    🔥 Urgente
                                </span>
                            </div>
                        `).join('');
                    }
                } catch (error) {
                    console.error('Error:', error);
                }
            }
            
            actualizarDatos();
            setInterval(actualizarDatos, 10000);
        </script>
    </body>
    </html>
    """

@app.get("/api/dashboard")
async def dashboard():
    session = SessionLocal()
    try:
        total_productos = session.query(Producto).filter(Producto.activo == True).count()
    finally:
        session.close()
    
    leads_calientes = [lead for lead in leads_db.values() if lead.get("urgente", False)]
    
    return {
        "total_productos": total_productos,
        "total_clientes": len(clientes_db),
        "leads_calientes": len(leads_calientes),
        "leads": leads_calientes[:10]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
