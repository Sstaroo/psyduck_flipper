from pyglet import *
from OpenGL.GL import *
import pyglet
import pyglet.gl as GL
import trimesh as tm
import numpy as np
import os
from pathlib import Path
from pyglet.window import key
import pymunk
import sys
from trimesh.path.polygons import projected
from trimesh.repair import fix_normals
import grafica.lighting_shaders as ls
from grafica import transformations as tr
import time
import threading
import trimesh as tm
from trimesh.path.polygons import projected
from trimesh.repair import fix_normals
import numpy as np
from random import random
import random
from pyglet.window import mouse
#import RectangleCollision




if sys.path[0] != "":
    sys.path.insert(0, "")

# una función auxiliar para cargar shaders
from grafica.utils import load_pipeline

from grafica.arcball import Arcball
from grafica.textures import texture_2D_setup

LIGHT_FLAT    = 0
LIGHT_GOURAUD = 1
LIGHT_PHONG   = 2

presion_A = False
presion_D = False





# Different shader programs for different lighting strategies
textureFlatPipeline = ls.SimpleTextureFlatShaderProgram()
textureGouraudPipeline = ls.SimpleTextureGouraudShaderProgram()
texturePhongPipeline = ls.SimpleTexturePhongShaderProgram()
lightingPipeline = ls.SimplePhongShaderProgram()

#Aquí me basé en el aux 2 :D
def frustum(left, right, bottom, top, near, far):
    r_l = right - left
    t_b = top - bottom
    f_n = far - near
    return np.array([
        [ 2 * near / r_l,
        0,
        (right + left) / r_l,
        0],
        [ 0,
        2 * near / t_b,
        (top + bottom) / t_b,
        0],
        [ 0,
        0,
        -(far + near) / f_n,
        -2 * near * far / f_n],
        [ 0,
        0,
        -1,
        0]], dtype = np.float32)

def perspective(fovy, aspect, near, far):

    halfHeight = np.tan(np.pi * fovy / 360) * near
    halfWidth = halfHeight * aspect
    return frustum(-halfWidth, halfWidth, -halfHeight, halfHeight, near, far)

def lookAt(eye, at, up):

    forward = (at - eye)
    forward = forward / np.linalg.norm(forward)

    side = np.cross(forward, up)
    side = side / np.linalg.norm(side)

    newUp = np.cross(side, forward)
    newUp = newUp / np.linalg.norm(newUp)

    return np.array([
            [side[0],       side[1],    side[2], -np.dot(side, eye)],
            [newUp[0],     newUp[1],   newUp[2], -np.dot(newUp, eye)],
            [-forward[0], -forward[1], -forward[2], np.dot(forward, eye)],
            [0,0,0,1]
        ], dtype = np.float32)

def translate(tx, ty, tz):

    return np.array([
        [1,0,0,tx],
        [0,1,0,ty],
        [0,0,1,tz],
        [0,0,0,1]], dtype = np.float32)
def scale(sx, sy, sz):
    
    return np.array([
        [sx,0,0,0],
        [0,sy,0,0],
        [0,0,sz,0],
        [0,0,0,1]], dtype = np.float32)

def rotationX(theta):

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    return np.array([
        [1,0,0,0],
        [0,cos_theta,-sin_theta,0],
        [0,sin_theta,cos_theta,0],
        [0,0,0,1]], dtype = np.float32)

def rotationY(theta):

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    return np.array([
        [cos_theta,0,sin_theta,0],
        [0,1,0,0],
        [-sin_theta,0,cos_theta,0],
        [0,0,0,1]], dtype = np.float32)

def rotationZ(theta):

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    return np.array([
        [cos_theta,-sin_theta,0,0],
        [sin_theta,cos_theta,0,0],
        [0,0,0,0],
        [0,0,0,1]], dtype = np.float32)

def load_pipeline(vertex_shader_path, fragment_shader_path):
    with open(vertex_shader_path) as f:
        vertex_source = f.read()
    with open(fragment_shader_path) as f:
        fragment_source = f.read()
    vertex_shader = pyglet.graphics.shader.Shader(vertex_source, "vertex")
    fragment_shader = pyglet.graphics.shader.Shader(fragment_source, "fragment")
    return pyglet.graphics.shader.ShaderProgram(vertex_shader, fragment_shader)

def setupMesh(file_path, tex_pipeline, notex_pipeline, scale, texture_path):
    asset = tm.load(file_path, force="scene")
    asset.rezero()
    asset = asset.scaled(scale / asset.scale)
    #print(file_path)
    #print(asset.bounds)

    vertex_lists = {}
    for object_id, object_geometry in asset.geometry.items():
        mesh = {}
        object_geometry.fix_normals(True)
        object_vlist = tm.rendering.mesh_to_vertexlist(object_geometry)
        n_triangles = len(object_vlist[4][1]) // 3

        # Verificar si object_geometry.visual tiene material y si tiene imagen
        has_texture = hasattr(object_geometry.visual, 'material') and object_geometry.visual.material.image is not None

        if has_texture:
            #print('has texture')
            mesh["pipeline"] = tex_pipeline
        else:
            #print('no texture')
            mesh["pipeline"] = notex_pipeline

        mesh["gpu_data"] = mesh["pipeline"].vertex_list_indexed(
            n_triangles, GL.GL_TRIANGLES, object_vlist[3]
        )
        mesh["gpu_data"].position[:] = object_vlist[4][1]
        mesh["gpu_data"].normal[:] = object_vlist[5][1]

        if has_texture:
            try:
                mesh["texture"] = texture_2D_setup(texture_path)
                uv_list = list(object_vlist[6][1])
                if len(uv_list) != len(mesh["gpu_data"].uv):
                    if len(uv_list) > len(mesh["gpu_data"].uv):
                        # Recortar la secuencia si es más larga
                        uv_list = uv_list[:len(mesh["gpu_data"].uv)]
                    elif len(uv_list) < len(mesh["gpu_data"].uv):
                        # Expandir la secuencia si es más corta
                        uv_list = uv_list + [0.0] * (len(mesh["gpu_data"].uv) - len(uv_list))
                mesh["gpu_data"].uv[:] = uv_list
            except ValueError as e:
                print(f"Error al cargar la textura para {object_id}: {e}")
                mesh["pipeline"] = notex_pipeline
                mesh["gpu_data"].color[:] = object_vlist[6][1]
        else:
            mesh["gpu_data"].color[:] = object_vlist[6][1]

        mesh['id'] = object_id[0:-4]
        vertex_lists[object_id] = mesh

    return vertex_lists

def texture_2D_setup(image_path):
    image = pyglet.image.load(image_path)
    texture = image.get_texture()
    GL.glBindTexture(GL.GL_TEXTURE_2D, texture.id)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    return texture

#----------------------------------------------------------------
#Utilidades
def distancia(v1, v2):
    return np.sqrt((v2[0] - v1[0])*(v2[0] - v1[0]) + (v2[1] -v1[1])*(v2[1] - v1[1]))

def magnitud(v1):
		return np.sqrt(v1[0] * v1[0] + v1[1] * v1[1])
def normalizar(v1):
		mag = magnitud(v1)
		if not (mag == 0 ):
			return (v1[0]/mag,v1[1]/mag)
def limits(v1, max_length):
        a = (0,0)
        squared_mag = magnitud(v1) * magnitud(v1)
        if squared_mag > (max_length * max_length):
            a[0] = v1[0]/np.sqrt(squared_mag)
            a[1] = v1[1]/np.sqrt(squared_mag)
            a[0] = v1[0] * max_length
            a[1] = v1[1] * max_length
            return a
#-----------------------------------------------------------------
#Estela para la pelota
# class Particulas:
#     def __init__(self, position, ttl):
#         self.position = (position[0], position[1])



#----------------------------------------------------------------
#DINAMICA DE PARTICULAS CON POKEBOLAS
pokeball_positions = []
class pokeball:
    def __init__(self, space):
        self.position = (random.randint(6,9)/10, random.randint(0, 5)/10)
        masa = 0.1
        radio = 0.1
        self.acc =(0,0)
        self.max_velocidad = 2
        self.max_largo = 1
        #self.velocity = (0,0)
        momento = pymunk.moment_for_circle(masa, 0, radio)
        self.pokeball_body = pymunk.Body(masa, momento)
        self.pokeball_body.position = self.position  # Posición inicial arbitraria
        self.pokeball_shape = pymunk.Circle(self.pokeball_body, radio, (0, 0))
        space.add(self.pokeball_body, self.pokeball_shape)
        pokeball_positions.append(self)


    def cohesion_mouse(self, mouse, k):
        #mouse tambien es un vector ESPERO
        vector = self.pokeball_body.position
        d = distancia(vector, mouse)
        return (k*(mouse[0]-vector[0])/d, k*(mouse[1]-vector[1])/d)
        
    
#----------------------------------------------------------------
class Controller:

    def __init__(self):
        self.scaleX = 1
        self.scaleY = 1
        self.scaleZ = 1
        self.currentColor = 0
        self.colors = [
            [1,0,0],[0,1,0],[0,0,1]
        ]
        self.rotating = False
        self.alphaX = 0
        self.alphaY = 0
        self.alphaZ = 0
        self.max_rotation = 45
        self.x = 0
        self.y = 0
        self.z = 0
        self.camera_theta = np.pi/4
        self.seguir_mouse = False
        self.p_mouse = (0,0)
        #self.cuenta = 0

        #Label
        self.label = 'Modo Libre'
        self.label_act = True
        self.musicaaa = False

        def startRotation(self):
            self.rotating = True
        def stopRotation(self):
            self.rotating = False

        #Primer estado de los Flippers
        self.flipper_izq_up = False
        self.flipper_der_up = False

        # Definir la variable para la vista superior
        self.upper_view = False

        # Definir la variable para el pipeline de la cámara
        self.camera_pipeline = None

    def scaleUp(self):
        self.scaleX += 0.3
        self.scaleY += 0.3
        self.scaleZ += 0.3
    def scaleDown(self):
        self.scaleX -= 0.3
        self.scaleY -= 0.3
        self.scaleZ -= 0.3
    def getColor(self):
        return self.colors[self.currentColor]
    def changeColor(self):
        self.currentColor = (self.currentColor+1)%len(self.colors)
    def startRotation(self):
        self.rotating = True
    def stopRotation(self):
        self.rotating = False
    def moveLeft(self):
        self.y -= 0.2
    def moveRight(self):
        self.y += 0.2
    


    def get_view_matrices(self):
        # Vista predeterminada
        view = lookAt(np.array([0, 0, 3]), np.array([0, 0, 0]), np.array([0, 1, 0]))
        projection = perspective(45, window.width / window.height, 0.1, 100)

        if self.upper_view:
            # Ajustar la posición de la cámara para inclinar 45 grados
            eye = np.array([0, -3 * np.cos(np.radians(45)), 3 * np.sin(np.radians(45))])
            at = np.array([0, 0, 0])
            up = np.array([0, 1, 0])

            view = lookAt(eye, at, up)
            projection = perspective(45, window.width / window.height, 0.1, 100)

        return view, projection

    def set_view_projection(self, pipeline):
        view, projection = self.get_view_matrices()
        pipeline['view'] = view.reshape(16, 1, order='F')
        pipeline['projection'] = projection.reshape(16, 1, order='F')

    def ChangeView(self):
        self.upper_view = not self.upper_view

    #Definimos funciones para controlar los Flippers.
    def rise_flipper_izq(self):
        self.flipper_izq_up = True

    def fall_flipper_izq(self):
        self.flipper_izq_up = False

    def rise_flipper_der(self):
        self.flipper_der_up = True

    def fall_flipper_der(self):
        self.flipper_der_up = False
    def cambiar_seguir(self):
        self.seguir_mouse = not self.seguir_mouse

    def cambiar_label(self, letra):
        self.label_act = not self.label_act

    def EnBoton(self):
        if (self.p_mouse[0]>0) and (self.p_mouse[0]<100) and (self.p_mouse[1]>0) and (self.p_mouse[1]<100):
            return True
        else:
            return False
  


#--------------------------------------------------------------------------
if __name__ == "__main__":

    window = pyglet.window.Window(960, 960)

    controller = Controller()

    tex_pipeline = load_pipeline(
        Path(os.path.dirname(__file__)) / "vertex_program.glsl",
        Path(os.path.dirname(__file__)) / "fragment_program.glsl"
    )

    notex_pipeline = load_pipeline(
        Path(os.path.dirname(__file__)) / "vertex_program_notex.glsl",
        Path(os.path.dirname(__file__)) / "fragment_program_notex2.glsl"
    )
    
    vertex_lists = {}
    vertex_lists = setupMesh('tarea3/Tablero.obj', tex_pipeline, notex_pipeline, 1.7,'tarea3/Texturas/Metal_Amarillo_Fuerte.jpeg' )
    vertex_lists.update(setupMesh('tarea3/Brazo.obj', tex_pipeline, notex_pipeline, 0.5,'tarea3/Texturas/Metal_Amarillo.jpeg' ))
    vertex_lists.update(setupMesh('tarea3/Ojo_Derecho.obj', tex_pipeline, notex_pipeline, 0.2,'tarea3/Texturas/Blanco.jpeg' ))
    vertex_lists.update(setupMesh('tarea3/Ojo_Izquierdo.obj', tex_pipeline, notex_pipeline, 0.2,'tarea3/Texturas/Blanco.jpeg' ))
    vertex_lists.update(setupMesh('tarea3/Pupila_Derecha.obj', tex_pipeline, notex_pipeline, 0.02,'tarea3/Texturas/Negro.jpg' ))
    vertex_lists.update(setupMesh('tarea3/Pupila_Izquierda.obj', tex_pipeline, notex_pipeline, 0.02,'tarea3/Texturas/Negro.jpg' ))
    vertex_lists.update(setupMesh('tarea3/Pico.obj', tex_pipeline, notex_pipeline, 0.65,'tarea3/Texturas/Amarillo_claro.jpg' ))
    vertex_lists.update(setupMesh('tarea3/Pata.obj', tex_pipeline, notex_pipeline, 0.3,'tarea3/Texturas/Amarillo_claro.jpg' ))
    vertex_lists.update(setupMesh('tarea3/Pelos.obj', tex_pipeline, notex_pipeline, 0.25,'tarea3/Texturas/Negro.jpg' ))
    vertex_lists.update(setupMesh('tarea3/Pelota.obj', tex_pipeline, notex_pipeline, 0.1, 'tarea3/Texturas/metalplata.png'))
    vertex_lists.update(setupMesh('tarea3/Pared_Izquierda.obj', tex_pipeline, notex_pipeline, 0.3, 'tarea3/Texturas/Metal_Amarillo.jpeg'))
    vertex_lists.update(setupMesh('tarea3/Pared_Derecha.obj', tex_pipeline, notex_pipeline, 0.3, 'tarea3/Texturas/Metal_Amarillo.jpeg'))
    vertex_lists.update(setupMesh('tarea3/pokeball_simple.obj', tex_pipeline, notex_pipeline, 0.3, 'tarea3/Texturas/HD-wallpaper-pokeball-background-for-abstract-pokeball.jpg'))
  #---------------------------------------------------------------------------------------------------------------------------------------------------------------------



    # Inicializar el espacio de Pymunk
    space = pymunk.Space()
    space.gravity = (0,-1)
#----------------------------------------------------------------    
    segment_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    space.add(segment_body)

#----------------------------------------------------------------    

    BALL_COLLISION_TYPE = 1
    BOTTOM_COLLISION_TYPE = 2
    COLLISION_TYPE_CIRCULO_ARRIBA = 3
 
    ball_mass = 0.1
    new_ball_radius = 0.05
    new_ball_moment = pymunk.moment_for_circle(ball_mass, 0, new_ball_radius)
    ball_body = pymunk.Body(ball_mass, new_ball_moment)
    ball_body.position = (1234,-0.5)  # Posición inicial arbitraria
    ball_shape = pymunk.Circle(ball_body, new_ball_radius, (0, 0))
    ball_shape.collision_type = BALL_COLLISION_TYPE
    space.add(ball_body, ball_shape)



#----------------------------------------------------------------
    # pokeball_mass = 0.1
    # new_pokeball_radius = 0.1
    # new_pokeball_moment = pymunk.moment_for_circle(pokeball_mass, 0, new_pokeball_radius)
    # pokeball_body = pymunk.Body(pokeball_mass, new_pokeball_moment)
    # pokeball_body.position = (0.6,0)  # Posición inicial arbitraria
    # pokeball_shape = pymunk.Circle(pokeball_body, new_pokeball_radius, (0, 0))
    # space.add(pokeball_body, pokeball_shape)
#----------------------------------------------------------------

# Tablero

    segment_body1 = pymunk.Body(body_type=pymunk.Body.STATIC)
    segment1 = pymunk.Segment(segment_body1, (-0.4, -0.82), (0.4, -0.82), 0.001)   # abajo
    segment1.collision_type = BOTTOM_COLLISION_TYPE

    segment_body2 = pymunk.Body(body_type=pymunk.Body.STATIC)
    segment2 = pymunk.Segment(segment_body2, (-0.4, 0.7), (0.4, 0.7), 0.001)     # arriba

    segment_body3 = pymunk.Body(body_type=pymunk.Body.STATIC)
    segment3 = pymunk.Segment(segment_body3, (0.4, -0.82), (0.4, 0.77), 0.001)     # der

    segment_body4 = pymunk.Body(body_type=pymunk.Body.STATIC)
    segment4 = pymunk.Segment(segment_body4, (-0.4, -0.82), (-0.4, 0.77), 0.001)   # izq

    segment_body5 = pymunk.Body(body_type=pymunk.Body.STATIC)
    segment5 = pymunk.Segment(segment_body5, (0.24, 0.7), (0.415, 0.29), 0.001)   # m

    segment_body6 = pymunk.Body(body_type=pymunk.Body.STATIC)
    segment6 = pymunk.Segment(segment_body6, (-0.43, 0.29), (-0.24, 0.68), 0.001)   # j

    segment_body7 = pymunk.Body(body_type=pymunk.Body.STATIC)
    segment7 = pymunk.Segment(segment_body7, (-0.24, 0.7), (-0.07, 0.83), 0.001)   # k

    segment_body8 = pymunk.Body(body_type=pymunk.Body.STATIC)
    segment8 = pymunk.Segment(segment_body8, (0.07, 0.83), (0.24, 0.7), 0.001)   # l

    space.add(segment_body1, segment_body2, segment_body3, segment_body4,segment_body5,segment_body6,segment_body7,segment_body8)
    space.add(segment1, segment2, segment3, segment4,segment5,segment6,segment7,segment8)

#-----------------------------------------------------------------------------------
    import math
    def create_oval_shape(body, width, height, offset=(0, 0), num_segments=20):
        vertices = []
        for i in range(num_segments):
            angle = i * (2 * math.pi) / num_segments
            x = (width / 2) * math.cos(angle) + offset[0]
            y = (height / 2) * math.sin(angle) + offset[1]
            vertices.append((x, y))
        return pymunk.Poly(body, vertices)

    ojosizq_mass = 1
    ojosizq_moment = pymunk.moment_for_poly(ojosizq_mass, [(0,0)])
    ojosizq_body = pymunk.Body(ojosizq_mass, ojosizq_moment, body_type=pymunk.Body.STATIC)
    ojosizq_body.position = (-0.12, 0.40)

    # Crear un óvalo con un ancho de 0.07 y un alto de 0.03
    ojosizq_shape = create_oval_shape(ojosizq_body, 0.115, 0.07)
    space.add(ojosizq_body, ojosizq_shape)


    ojosder_mass = 1
    ojosder_moment = pymunk.moment_for_poly(ojosder_mass, [(0,0)])
    ojosder_body = pymunk.Body(ojosder_mass, ojosder_moment, body_type=pymunk.Body.STATIC)
    ojosder_body.position = (0.12, 0.40)

    # Crear un óvalo con un ancho de 0.07 y un alto de 0.03
    ojosder_shape = create_oval_shape(ojosder_body, 0.115, 0.07)
    space.add(ojosder_body, ojosder_shape)


    def create_half_oval_shape(body, width, height, offset=(0, 0), num_segments=20):
        vertices = []
        for i in range(num_segments // 2, num_segments):
            angle = i * (2 * math.pi) / num_segments
            x = (width / 2) * math.cos(angle) + offset[0]
            y = (height / 2) * math.sin(angle) + offset[1]
            vertices.append((x, y))
        # Adding the endpoints to close the shape
        vertices.append((width / 2, 0 + offset[1]))
        vertices.append((-width / 2, 0 + offset[1]))
        return pymunk.Poly(body, vertices)

    pico_mass = 1
    pico_moment = pymunk.moment_for_poly(pico_mass, [(0,0)])
    pico_body = pymunk.Body(pico_mass, pico_moment, body_type=pymunk.Body.STATIC)
    pico_body.position = (0, -0.05)

    # Crear la mitad inferior de un óvalo con un ancho de 0.115 y un alto de 0.07
    pico_shape = create_half_oval_shape(pico_body, 0.33, 0.5)
    space.add(pico_body, pico_shape)


    segment10 = pymunk.Segment(segment_body, (-0.17, 0), (0, 0.2), 0.001)     # a c
    segment11 = pymunk.Segment(segment_body, (0, 0.2), (0.17, 0), 0.001)     # c b
    
    segment12 = pymunk.Segment(segment_body, (-0.17, 0), (-0.13,0.16), 0.001)   # a d
    segment13 = pymunk.Segment(segment_body, (-0.13,0.16), (-0.03, 0.17), 0.001)     # d e

    segment14 = pymunk.Segment(segment_body, (0.17, 0), (0.13,0.16), 0.001)     # b g
    segment15 = pymunk.Segment(segment_body, (0.13,0.16), (0.03, 0.17), 0.001)   # g f

    space.add(segment10, segment11, segment12,segment13, segment14, segment15)


    segment16 = pymunk.Segment(segment_body, (-0.38, -0.55), (-0.3, -0.567), 0.001)     # izq
    segment17 = pymunk.Segment(segment_body, (-0.3, -0.567), (-0.17,-0.6), 0.001)     # der
    segment18 = pymunk.Segment(segment_body, (-0.17,-0.6), (-0.1,-0.72), 0.001)   # arriba

    segment19 = pymunk.Segment(segment_body, (0.38, -0.55), (0.3, -0.567), 0.001)     # izq
    segment20 = pymunk.Segment(segment_body, (0.3, -0.567), (0.17,-0.6), 0.001)     # der
    segment21 = pymunk.Segment(segment_body, (0.17,-0.6), (0.1,-0.72), 0.001)   # arriba

    space.add(segment16, segment17, segment18,segment19, segment20, segment21)

    segment22 = pymunk.Segment(segment_body, (-0.4, -0.45), (-0.3, -0.567), 0.001) 
    segment23 = pymunk.Segment(segment_body, (0.4, -0.45), (0.3, -0.567), 0.001) 
    space.add(segment22, segment23)

    segment24 = pymunk.Segment(segment_body, (0.4, -0.6), (1.4, -0.6), 0.001) 
    segment25 = pymunk.Segment(segment_body, (1.25, -0.6), (1.25, 1.7), 0.001)
    space.add(segment24,segment25)
#----------------------------------------------------------------------------------
# Flipper

    pataizq_shape = pymunk.Poly(None, ((0.1, 0), (0.14, 0.015), (0.1, 0.04), (-0.12, 0.04)))
    pataizq_body = pymunk.Body(1, pymunk.moment_for_poly(1, pataizq_shape.get_vertices(), (-0.2, 0)), body_type=pymunk.Body.KINEMATIC)
    pataizq_shape.body = pataizq_body
    pataizq_body.angle = 0
    pataizq_body.position = (-0.31, -0.65)
    pataizq_body.center_of_gravity = (-0.095,0.01)
    space.add(pataizq_body, pataizq_shape)

    patader_shape = pymunk.Poly(None, ((-0.1, 0), (-0.14, 0.015), (-0.1, 0.04), (0.12, 0.04)))
    patader_body = pymunk.Body(1, pymunk.moment_for_poly(1, patader_shape.get_vertices(), (0.2, 0)), body_type=pymunk.Body.KINEMATIC)
    patader_shape.body = patader_body
    patader_body.angle = 0
    patader_body.center_of_gravity = (0.095,0.015)
    patader_body.position = (0.225, -0.65)
    space.add(patader_body, patader_shape)

    # Diccionario para almacenar los cuerpos
    bodies = {}
    bodies['pelota'] = ball_body
    # bodies['pokeball'] = pokeball_body
    bodies['Ojo_Izquierdo'] = ojosizq_body
    bodies['Ojo_Derecho'] = ojosder_body
    bodies['Pico'] = pico_body
    bodies['PataIzq'] = pataizq_body
    bodies['PataDer'] = patader_body

    arcball = Arcball(
        np.identity(4),
        np.array((960, 960), dtype=float),
        1.5,
        np.array([0.0, 0.0, 0.0]),
    )

  
    # Crear Luces
    # Tercera Luz
    light3_color = np.array([0.0, 0.0, 1], dtype=np.float32)  # Azul

    def setup_light3(pipeline):
        # Configurar la luz 3 con su color y posición
        pipeline['light3_enabled'] = True
        pipeline['light3_color'] = light3_color
        pipeline["light3_position"] = np.array([np.sin(time.time()), np.cos(time.time()), 0])

    # Estado de la luz
    light2_enabled = False
    #Agregmos Meme
    imagen = pyglet.image.load('tarea3/emoji.png')
    meme = pyglet.sprite.Sprite(imagen, x=240, y=600)
    def handle_collision(arbiter, space, data, pipeline):
        global light2_enabled  # Referenciar la variable global para el estado de la luz
        #print("¡Colisión detectada entre la pelota y la pared de abajo!")
        
        # Resetear la posición de la pelota
        ball_body.position = (0, 0)
        
        # Mover la luz a una nueva posición (ejemplo)
        new_light_position = np.array([0, -5, 0])  # Define la nueva posición para la luz
        pipeline["light2_position"] = new_light_position
        
        # Activar la segunda luz
        activar_luz(pipeline)
        #meme.draw()
        # Programar la desactivación de la luz después de 3 segundos
        pyglet.clock.schedule_once(lambda dt: desactivar_luz(pipeline), 1)
        #pyglet.clock.schedule_once(lambda dt: meme.draw(), 3)
        return True  # Retorna True para permitir que Pymunk maneje la resolución de colisiones

    collision_handler = space.add_collision_handler(BALL_COLLISION_TYPE, BOTTOM_COLLISION_TYPE)
    collision_handler.begin = handle_collision

    def activar_luz(pipeline):
        global light2_enabled  # Referenciar la variable global
        pipeline["light2_enabled"] = True  # Activa la segunda luz
        meme.draw()
        light2_enabled = True
        #print('luz activa')

    def desactivar_luz(pipeline):
        global light2_enabled  # Referenciar la variable global
        pipeline["light2_enabled"] = False  # Desactiva la segunda luz
        light2_enabled = False
        
#--------------------------------------------
    # def FUNCION_QLIA():
    #     pataizq_body.angular_velocity = 10

#--------------------------------------------
    controller.camera_pipeline = tex_pipeline
    musica = pyglet.resource.media('song.mp3', streaming= False)
    
    
    def set_angular_velocity_izq(velocity):
        pataizq_body.angular_velocity = velocity

    # Función para establecer la velocidad angular del flipper derecho
    def set_angular_velocity_der(velocity):
        patader_body.angular_velocity = velocity
    
    @window.event

    def on_key_press(symbol, modifier):


        if(key.A == symbol):
            #controller.levantar_flipper_izquierdo()
            pataizq_body.angular_velocity = 10
            threading.Timer(0.05, set_angular_velocity_izq, [0]).start()

        if(key.D == symbol):
            patader_body.angular_velocity = -10
            threading.Timer(0.05, set_angular_velocity_der, [0]).start()

        # Cambio de vista
        if(key.C == symbol):
            #print('Cambiar vista')
            controller.ChangeView()

        if(key.SPACE == symbol):
            # Aplicar una velocidad inicial hacia abajo a la pelota
            ball_body.position = (0.37, -0.3)
            ball_body.velocity = (0, 1.4)
            #print("POR FAVOR MUEVETE PELOTA QLIA")
        if(key.L == symbol):
            pokeball(space)
            #controller.cuenta +=1

        if(key.V == symbol):
            controller.cambiar_seguir()
            controller.cambiar_label('v')

        if(key.M == symbol):
            musica.play()
           
            #print('hola')
        # if(key.B == symbol):
        #     ball_body.velocity = (-0.25, 0)

        # if(key.N == symbol):
        #     ball_body.velocity = (0, -0.4)

        # if(key.J == symbol):
        #     ball_body.velocity = (0, 0.4)


    @window.event

    def on_key_release(symbol, modifier):

        if(key.A == symbol):
            pataizq_body.angle = 0
            #print('Patita Derecha')
        if(key.D == symbol):
            patader_body.angle = 0
            #print('Patita Izquierda')

    @window.event

    def on_mouse_motion(x,y,dx,dy):
        controller.p_mouse = (x,y)


    blue_label = pyglet.text.Label(controller.label, font_size=30, font_name='Times New Roman', x= 100, y=940, anchor_x='center', anchor_y='center')
#-----------------------------------------------------------------------------
    #Haremos un update para que le juego nos detecte la presión de A y D SOLO UNA VEZ
    # angulo_izq = pataizq_body.angle
    # def update(dt):
    #     global presion_A
    #     global presion_D
    #     if presion_A and angulo_izq < 1:
    #         pataizq_body.angular_velocity = 10
    #         presion_A = False



#-----------------------------------------------------------------------------
    # GAME LOOP
    @window.event
    def on_draw():
        global max_h
        GL.glClearColor(0.1, 0.1, 0.1, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        GL.glLineWidth(1)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        
        window.clear()
        #DINAMICA DE POKEBOLAS
        if controller.seguir_mouse:
            for pp in pokeball_positions:
                #print(pp.pokeball_body.position)
                #print(' vs ') 
                #print(controller.p_mouse)
                x,y = controller.p_mouse[0], controller.p_mouse[1]
                #factores para el cambio de coordenadas
                fx, fy = 0.002625, 0.002625
                mouse_posiciones = ((x-480)*fx,(y-480)*fy)
                #print(mouse_posiciones)
                pp.pokeball_body.velocity = (0,0)
                pp.pokeball_body.velocity = pp.cohesion_mouse(mouse_posiciones, 0.5)

        if(controller.rotating):
            controller.alphaX += 0.01
        

        
            #print(ball_body.position) #devuelve vectores en 2D de la masa de la pelota

        glUseProgram(lightingPipeline.shaderProgram)

        projection = perspective(45, float(960)/float(960), 0.1, 100)
        camX = 3 * np.sin(controller.camera_theta)
        camY = 3 * np.cos(controller.camera_theta)

        viewPos = np.array([camX,camY,2])

        # Configuración de los uniformes de iluminación
        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "La"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "Ld"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "Ls"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "Ka"), 0.2, 0.2, 0.2)
        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "Kd"), 0.9, 0.9, 0.9)
        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "Ks"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "lightPosition"), 0, 0, 10)
        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "viewPosition"), viewPos[0], viewPos[1], viewPos[2])
        glUniform1ui(glGetUniformLocation(lightingPipeline.shaderProgram, "shininess"), 1000)
        glUniform1f(glGetUniformLocation(lightingPipeline.shaderProgram, "constantAttenuation"), 0.0001)
        glUniform1f(glGetUniformLocation(lightingPipeline.shaderProgram, "linearAttenuation"), 0.01)
        glUniform1f(glGetUniformLocation(lightingPipeline.shaderProgram, "quadraticAttenuation"), 0.01)
        glUniformMatrix4fv(glGetUniformLocation(lightingPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(lightingPipeline.shaderProgram, "view"), 1, GL_TRUE, viewPos)
        hola = time.time()
        if True:
            space.step(0.02)

        if controller.label_act:
            blue_label.draw()
        if controller.label_act == False:
            pyglet.text.Label('Modo Cohesión', font_size=30, font_name='Times New Roman', x= 132, y=940, anchor_x='center', anchor_y='center').draw()

        pyglet.text.Label('[V] para cambiar de modo', font_size=14, font_name='Times New Roman', x= 100, y=910, anchor_x='center', anchor_y='center').draw()

        pyglet.text.Label('[L] para agregar pokeballs', font_size=14, font_name='Times New Roman', x= 100, y=880, anchor_x='center', anchor_y='center').draw()

        pyglet.text.Label('[M] para poner música', font_size=14, font_name='Times New Roman', x= 86, y=850, anchor_x='center', anchor_y='center').draw()

        pyglet.text.Label('SPACE para iniciar el Flipper', font_size=14, font_name='Times New Roman', x= 113, y=820, anchor_x='center', anchor_y='center').draw()

            #pokeball_body.velocity = (0,0)
            #print(pokeball_body.position)
            # for pp in pokeball_positions:
            #print(p_mouse)
        
            


        for object_id, object_geometry in vertex_lists.items():
            pipeline = object_geometry["pipeline"]
            pipeline.use()
            controller.set_view_projection(pipeline)
            pipeline["light_position"] = np.array([0, 0, -0.15])
            #pipeline["light_position"] = np.array([0.1, 0, 0.15])

            # Registra el manejador de colisiones con el pipeline
            collision_handler = space.add_collision_handler(BALL_COLLISION_TYPE, BOTTOM_COLLISION_TYPE)
            collision_handler.begin = lambda arbiter, space, data: handle_collision(arbiter, space, data, pipeline)

            # Configurar la transformación para cada objeto
            transform_matrix = translate(controller.x, controller.y, controller.z) @ rotationX(controller.alphaX) @ scale(controller.scaleX, controller.scaleY, controller.scaleZ)

            if object_id == 'Brazo.obj':
                transform_matrix = translate(-0.65, 0.1,0) @ rotationX(controller.alphaX) @ rotationZ(np.cos(hola)-(0.7*np.cos(hola))-0.1)

            if object_id == 'Ojo_Izquierdo.obj':
                x,y = ojosizq_body.position[0], ojosizq_body.position[1]
                transform_matrix = translate(x,y,0)
            #     transform_matrix = translate(0, 0.17, 0)

            if object_id == 'Ojo_Derecho.obj':
                x,y = ojosder_body.position[0], ojosder_body.position[1]
                transform_matrix = translate(x,y,0)
            
            if object_id == 'Pelos.obj':
                transform_matrix = translate(0, 0.85, 0) @ rotationX(controller.alphaX) @rotationY(hola * np.pi)

            if object_id == 'Pico.obj':
                x,y = pico_body.position[0], pico_body.position[1]
                transform_matrix = translate(x,y,-0.015)

            if object_id == 'Pupila_Derecha.obj':
                transform_matrix = translate(0.11,0.425,0.02)

            if object_id == 'Pupila_Izquierda.obj':
                transform_matrix = translate(-0.11,0.425,0.02) 
            
            # if object_id == 'Pared_Izquierda.obj':
            #     transform_matrix = translate(-0.25,-0.57,0.02) @scale(1,0.5,1)
            # if object_id == 'Pared_Derecha.obj':
            #     transform_matrix = translate(0,0,0.02)

            if object_id == 'Pata.obj':
                x,y = pataizq_body.position[0], pataizq_body.position[1]
                z,w = patader_body.position[0], patader_body.position[1]
                alpha = pataizq_body.angle + np.pi
                betha = patader_body.angle
                

                # Definir las posiciones de los Flippers
                flipper_positions = [
                    (z, w, 0),  # Primera posición del Flipper der
                    (x, y, 0)    # Segunda posición del Flipper izq
                ]
                # Iterar sobre las posiciones y renderizar cada Flipper
                for i, pos in enumerate(flipper_positions):
                    if i == 1:
                        # Rotar solo la primera instancia
                        transform_matrix = translate(x ,y,0)@ rotationZ(alpha)@scale(1.3,1,123)
                    else:
                        # No rotar la segunda instancia
                        transform_matrix = translate(z ,w,0)@ rotationZ(betha)@scale(1.3,1,123)
                    
                    pipeline["transform"] = transform_matrix.reshape(16, 1, order='F')
                    # Asignar texturas si es necesario
                    if "texture" in object_geometry:
                        GL.glBindTexture(GL.GL_TEXTURE_2D, object_geometry["texture"].id)
                    else:
                        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
                    object_geometry["gpu_data"].draw(pyglet.gl.GL_TRIANGLES) 
 #@scale(1.3,1,1)
        
            if object_id == 'Pelota.obj':
                x,y = ball_body.position[0], ball_body.position[1]
                j,k = ball_body.velocity[0], ball_body.velocity[1]
                transform_matrix = translate(x,y,0) @rotationX(np.cos(k)) @rotationY(np.cos(j)) 

            if object_id == 'pokeball_simple.obj':
                # f,g = pokeball_body.position[0], pokeball_body.position[1]
                # vx = pokeball_body.velocity[0]
                # #gamma = pokeball_body.angle 
                # transform_matrix = translate(f,g,0)@rotationX(np.pi ) @ rotationY(np.pi/2) 
                if pokeball_positions == []:
                    transform_matrix = translate(1234566, 1234, 0)@rotationX(np.pi)@ rotationY(np.pi/2)
                    pipeline["transform"] = transform_matrix.reshape(16, 1, order='F')
                    if "texture" in object_geometry:
                            GL.glBindTexture(GL.GL_TEXTURE_2D, object_geometry["texture"].id)
                    else:
                            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
                    object_geometry["gpu_data"].draw(pyglet.gl.GL_TRIANGLES)
                else:
                    for pp in pokeball_positions:
                        transform_matrix = translate(pp.pokeball_body.position[0], pp.pokeball_body.position[1], 0)@rotationX(np.pi)@ rotationY(np.pi/2)
                        pipeline["transform"] = transform_matrix.reshape(16, 1, order='F')
                        # Asignar texturas si es necesario
                        if "texture" in object_geometry:
                            GL.glBindTexture(GL.GL_TEXTURE_2D, object_geometry["texture"].id)
                        else:
                            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
                        object_geometry["gpu_data"].draw(pyglet.gl.GL_TRIANGLES)


            
            pipeline["transform"] = transform_matrix.reshape(16, 1, order='F')

#--------------------------------------------------------------------------------------------------            
            ''' Asignar texturas '''
            if "texture" in object_geometry:
                GL.glBindTexture(GL.GL_TEXTURE_2D, object_geometry["texture"].id)
            else:
                GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

            object_geometry["gpu_data"].draw(pyglet.gl.GL_TRIANGLES)  

            # Función para activar la tercera luz (azul)
            setup_light3(pipeline)          

    #QUE SE HAGA EL LOOP!!!!
    #pyglet.clock.schedule_interval(update, 1/120)
    pyglet.app.run()

