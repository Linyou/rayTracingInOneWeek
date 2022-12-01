import numpy as np
from numpy.core.arrayprint import format_float_scientific

import taichi as ti

ti.init(arch=ti.vulkan)  # Try to run on GPU

infinity = np.Inf
pi = np.pi

samples_per_pixel = 100

@ti.func
def unit_vector(v):
    return v / ti.sqrt(v.dot(v))

@ti.func
def degrees_to_radians(degrees):
    return degrees * pi / 180.0

@ti.func
def clamp(x, min, max):
    if x < min:
        x = min
    if x > max:
        x = max
    return x

@ti.func
def near_zero(v):
    s = 1e-8
    v_abs = ti.abs(v)
    return (v_abs[0] < s) and (v_abs[1] < s) and (v_abs[2] < s)

@ti.func
def reflect(v, n):
    return v - 2 * v.dot(n) * n

@ti.func
def write_color(out: ti.template(), i, j, pixel_color, samples_per_pixel):
    r = pixel_color[0]
    g = pixel_color[1]
    b = pixel_color[2]

    scale = 1.0 / samples_per_pixel

    r = ti.sqrt(r * scale)
    g = ti.sqrt(g * scale)
    b = ti.sqrt(b * scale)

    out[i, j] = ti.Vector([
        clamp(r, 0.0, 0.999),
        clamp(g, 0.0, 0.999),
        clamp(b, 0.0, 0.999)
    ])


@ti.func
def random_in_unit_sphere():

    p = ti.Vector([
        ti.random(ti.f32), 
        ti.random(ti.f32), 
        ti.random(ti.f32)
    ]) * 2.0 - ti.Vector([1.0,1.0,1.0])
    length_squared = p.dot(p)
    while length_squared >= 1.0:
        p = ti.Vector([
            ti.random(ti.f32), 
            ti.random(ti.f32), 
            ti.random(ti.f32)
        ]) * 2.0 - ti.Vector([1.0,1.0,1.0])
        length_squared = p.dot(p)

    return p

@ti.func
def random_in_hemisphere(normal):

    in_unit_sphere = random_in_unit_sphere()
    out = -in_unit_sphere
    if in_unit_sphere.dot(normal) > 0.0:
        out = in_unit_sphere

    return out



# Hit record
class HitRecord:
    def __init__(self, res):
        self.record = ti.Struct.field({
            "p": ti.types.vector(3, ti.f32),
            "normal": ti.types.vector(3, ti.f32),
            "t": ti.f32,
            "front_face": ti.f32,
        }, shape=res)

    @ti.func
    def set_face_normal(self, i, j, outward_normal):
        if ray_holder.rays[i,j].direction.dot(outward_normal) < 0:
            self.record[i, j].front_face = 1.0
        else:
            self.record[i, j].front_face = -1.0

        if self.record[i, j].front_face > 0:
            self.record[i, j].normal = outward_normal
        else:
            self.record[i, j].normal = -outward_normal
    

# Ray
class Ray:
    def __init__(self, res):
        self.rays = ti.Struct.field({
                        "origin": ti.types.vector(3, ti.f32),
                        "direction": ti.types.vector(3, ti.f32),
                    }, shape=res)

    @ti.func
    def at(self, i, j, t):
        return self.rays[i,j].origin + t * self.rays[i,j].direction


# Hittable object
# Shpere
class Shpere:
    def __init__(self, descriptor):
        vec3f = ti.types.vector(3, ti.f32)
        shpere_field = ti.types.struct(
        center=vec3f, raduis=ti.f32,
        )

        self.size = len(descriptor)
        self.content = shpere_field.field(shape=(self.size,))

        for i, obj in enumerate(descriptor):
            self.content[i].center = ti.Vector(obj["center"])
            self.content[i].raduis = obj["raduis"]

    
    @ti.func
    def hit(self, index, i, j, t_min, t_max):
        # assume that is hit by ray
        hitting = True

        ray_origin = ray_holder.rays[i,j].origin
        ray_direction = ray_holder.rays[i,j].direction 
        oc = ray_origin - self.content[index].center
        a = ray_direction.dot(ray_direction)
        half_b = oc.dot(ray_direction)
        c = oc.dot(oc) - self.content[index].raduis*self.content[index].raduis
        discriminant = half_b*half_b - a*c
        sqrtd = ti.sqrt(discriminant)

        if discriminant < 0:
            hitting = False

        # Find the nearest root that lies in the acceptable range.
        root = (-half_b - sqrtd) / a
        if (root < t_min or t_max < root):
            root = (-half_b + sqrtd) / a
            if (root < t_min or t_max < root):
                hitting = False
        
        if hitting == True:
            hit_holder.record[i, j].t = root
            hit_holder.record[i, j].p = ray_holder.at(i, j, root)
            outward_normal = (hit_holder.record[i, j].p - self.content[index].center) / self.content[index].raduis
            hit_holder.set_face_normal(i, j, outward_normal)


        return hitting


# Hittable list
class HittableList:
    def __init__(self):
        self.shape = []

    @ti.func
    def hit(self, i, j, t_min, t_max):
        hit_anything = False
        closest_so_far = t_max

        # for shpere
        for index in range(shpere.size):
            if shpere.hit(index, i, j, t_min, closest_so_far):
                hit_anything = True
                closest_so_far = hit_holder.record[i, j].t

        return hit_anything

    def add(self, descriptor):
        for obj in descriptor:
            self.shape.append(obj)

    def clear(self):
        self.shpere_num = 0


# Camera
class Camera:
    def __init__(self):

        aspect_ratio = 16.0 / 9.0
        image_width = 1280
        image_height = int(image_width / aspect_ratio) 
        viewport_height = 2.0
        viewport_width = aspect_ratio * viewport_height
        focal_length = 1.0

        origin = np.array([0, 0, 0]).astype(np.float32)
        horizontal = np.array([viewport_width, 0, 0]).astype(np.float32)
        vertical = np.array([0, viewport_height, 0]).astype(np.float32)
        lower_left_corner = origin - horizontal/2 - vertical/2 - np.array([0, 0, focal_length]).astype(np.float32)

        self.origin = ti.Vector(origin.tolist())
        self.horizontal = ti.Vector(horizontal.tolist())
        self.vertical = ti.Vector(vertical.tolist())
        self.lower_left_corner = ti.Vector(lower_left_corner.tolist())
        self.res = (image_width, image_height)
        self.samples_per_pixel = 100

    @ti.func
    def set_ray(self, i, j, u, v):
        ray_holder.rays[i, j].origin = camera.origin
        ray_holder.rays[i, j].direction = camera.lower_left_corner + u * camera.horizontal + v * camera.vertical - camera.origin


max_depth = 50

camera = Camera()

# hit_holder = HitRecord(camera.res)
hit_holder = HitRecord(camera.res)
ray_holder = Ray(camera.res)

world = HittableList()

shpere_descriptor = []
shpere_descriptor.append({"shape_name": "shpere", "center": [0,0,-1], "raduis": 0.5})
shpere_descriptor.append({"shape_name": "shpere", "center": [0,-100.5,-1], "raduis": 100})

shpere = Shpere(shpere_descriptor)
world.add(shpere_descriptor)

image_width = camera.res[0]
image_height = camera.res[1]
samples_per_pixel = camera.samples_per_pixel

# Render
final_pixels = ti.Vector.field(3, dtype=float, shape=camera.res)

@ti.kernel
def render(reflect_method: ti.i32):
    for i, j in ti.ndrange((0, image_width), (0, image_height)):

        pixel_color = ti.Vector([0.0, 0.0, 0.0])
        for s in range(samples_per_pixel):
            u = (i + ti.random(ti.f32)) / (image_width - 1)
            v = (j + ti.random(ti.f32)) / (image_height-1)

            # ray
            camera.set_ray(i, j, u, v)

            color = ti.Vector([0.0,0.0,0.0])
            cur_attenuation = 1.0
            for d in range(max_depth):
                hitting = world.hit(i, j, 0.001, infinity)
                if hitting:
                    # 1.random in unit sphere
                    random_unit = hit_holder.record[i, j].normal + random_in_unit_sphere()
                    if reflect_method == 1:
                        # 2.True Lambertian Reflection
                        random_unit = hit_holder.record[i, j].normal + unit_vector(random_in_unit_sphere())  
                    if reflect_method == 2:
                        # 3.hemispherical scattering
                        random_unit = random_in_hemisphere(hit_holder.record[i, j].normal) 

                    target = hit_holder.record[i, j].p + random_unit
                    cur_attenuation *= 0.5
                    ray_holder.rays[i, j].origin = hit_holder.record[i, j].p
                    ray_holder.rays[i, j].direction = target - hit_holder.record[i, j].p
                else:
                    unit_direction = unit_vector(ray_holder.rays[i, j].direction)
                    t = 0.5*(unit_direction[1] + 1.0)
                    color = ((1.0-t)*ti.Vector([1.0, 1.0, 1.0]) + t*ti.Vector([0.5, 0.7, 1.0])) * cur_attenuation
                    break

            pixel_color += color
        write_color(final_pixels, i, j, pixel_color, samples_per_pixel)


window = ti.ui.Window("Taichi RayTracer", camera.res, vsync=True)
canvas = window.get_canvas()

reflect_select = 0
pause = False
while window.running:
    for e in window.get_events(ti.ui.PRESS):
        if e.key in [ti.ui.ESCAPE]:
            exit()
        elif e.key == 'p':
            pause = not pause
            print(pause)
    window.GUI.begin("reflect method", 0.05, 0.05, 0.15, 0.2)
    window.GUI.text("select reflect method")
    if window.GUI.button("in-unit-sphere"):
        reflect_select = 0
    if window.GUI.button("on-unit-sphere"):
        reflect_select = 1
    if window.GUI.button("in-hemis-phere"):
        reflect_select = 2
    window.GUI.end()
    if pause == False:
        render(reflect_select)
    # print(final_pixels)
    canvas.set_image(final_pixels)
    window.show()
