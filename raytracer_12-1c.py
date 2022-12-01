import numpy as np
from numpy.core.arrayprint import format_float_scientific

import taichi as ti

ti.init(arch=ti.vulkan)  # Try to run on GPU

infinity = np.Inf
pi = np.pi


@ti.pyfunc
def field1D_to_vector(v):
    u = v[None]
    return ti.Vector([u[0], u[1], u[2]])

@ti.pyfunc
def angle_cosine(a, b):
    return a.dot(b)/(a.norm() * b.norm())


@ti.pyfunc
def unit_vector(v):
    return v / ti.sqrt(v.dot(v))

@ti.func
def clamp(x, min, max):
    if x < min:
        x = min
    if x > max:
        x = max
    return x


@ti.pyfunc
def degrees_to_radians(d):
    return d * pi / 180.0


@ti.pyfunc
def radians_to_degrees(r):
    return r * 180.0 / pi


@ti.func
def near_zero(v):
    s = 1e-8
    v_abs = ti.abs(v)
    return (v_abs[0] < s) and (v_abs[1] < s) and (v_abs[2] < s)

@ti.func
def reflect(v, n):
    return v - 2 * v.dot(n) * n

@ti.func
def refract(uv, n, etai_over_etat):
    cos_theta = ti.min((-uv).dot(n), 1.0)
    r_out_perp = etai_over_etat * (uv + cos_theta*n)
    r_out_parallel = -ti.sqrt(ti.abs(1.0 - r_out_perp.dot(r_out_perp))) * n
    return r_out_perp + r_out_parallel

@ti.func
def write_color(out: ti.template(), i, j, pixel_color, samples_per_pixel, cnt):
    r = pixel_color[0]
    g = pixel_color[1]
    b = pixel_color[2]

    scale = 1.0 / samples_per_pixel

    r = ti.sqrt(r * scale)
    g = ti.sqrt(g * scale)
    b = ti.sqrt(b * scale)

    radience[i, j] += ti.Vector([
        clamp(r, 0.0, 0.999),
        clamp(g, 0.0, 0.999),
        clamp(b, 0.0, 0.999)
    ])

    out[i, j] = radience[i, j] / ti.cast(cnt, float)


@ti.func
def get_random_vector3_length():

    p = ti.Vector([
        ti.random(ti.f32), 
        ti.random(ti.f32), 
        ti.random(ti.f32)
    ]) * 2.0 - ti.Vector([1.0,1.0,1.0])
    length_squared = p.dot(p)
    
    return p, length_squared

@ti.func
def get_random_vector2_length():

    p = ti.Vector([
        ti.random(ti.f32), 
        ti.random(ti.f32)
    ]) * 2.0 - ti.Vector([1.0,1.0])
    length_squared = p.dot(p)
    
    return p, length_squared


@ti.func
def random_in_unit_disk():

    p, length_squared = get_random_vector2_length()

    while length_squared >= 1.0:
        p, length_squared = get_random_vector2_length()

    return p

@ti.func
def random_in_unit_sphere():

    p, length_squared = get_random_vector3_length()

    while length_squared >= 1.0:
        p, length_squared = get_random_vector3_length()

    return p


@ti.func
def random_in_hemisphere(normal):

    in_unit_sphere = random_in_unit_sphere()
    out = -in_unit_sphere
    if in_unit_sphere.dot(normal) > 0.0:
        out = in_unit_sphere

    return out

# Material
# lambertian
class Lambertian:
    def __init__(self, albedo):
        self.albedo = albedo

    @ti.func
    def scatter(self, i, j, attenuation: ti.template()):

        random_unit = hit_holder.record[i, j].normal + random_in_unit_sphere()

        if near_zero(random_unit):
            random_unit = hit_holder.record[i, j].normal 

        ray_holder.rays[i, j].origin = hit_holder.record[i, j].p
        ray_holder.rays[i, j].direction = random_unit

        attenuation = self.albedo

        return True

# metal
class Metal:
    def __init__(self, albedo, fuzz):
        self.albedo = albedo
        if fuzz > 1.0:
            fuzz = 1.0
        self.fuzz = fuzz

    @ti.func
    def scatter(self, i, j, attenuation: ti.template()):

        reflected = reflect(unit_vector(ray_holder.rays[i, j].direction), hit_holder.record[i, j].normal)

        ray_holder.rays[i, j].origin = hit_holder.record[i, j].p
        ray_holder.rays[i, j].direction = reflected + self.fuzz*random_in_unit_sphere()

        attenuation = self.albedo

        did_reflected = reflected.dot(hit_holder.record[i, j].normal) > 0
        return did_reflected

# dielectric
class Dielectric:
    def __init__(self, ir):
        self.ir = ir

    @ti.func
    def scatter(self, i, j, attenuation: ti.template()):

        attenuation = ti.Vector([1.0, 1.0, 1.0])
        
        refraction_ratio = self.ir
        if hit_holder.record[i, j].front_face == 1.0:
            refraction_ratio = 1.0/self.ir

        unit_direction = unit_vector(ray_holder.rays[i, j].direction)
        cos_theta = ti.min((-unit_direction).dot(hit_holder.record[i, j].normal), 1.0)
        sin_theta = ti.sqrt(1.0 - cos_theta*cos_theta)

        cannot_refract = refraction_ratio * sin_theta > 1.0

        direction = ti.Vector([1.0, 1.0, 1.0])
        did_reflectance = self.reflectance(cos_theta, refraction_ratio) > ti.random(ti.f32)
        # did_reflectance = False
        if cannot_refract or did_reflectance:
            direction = reflect(unit_direction, hit_holder.record[i, j].normal)
        else:
            direction = refract(unit_direction, hit_holder.record[i, j].normal, refraction_ratio)


        ray_holder.rays[i, j].origin = hit_holder.record[i, j].p
        ray_holder.rays[i, j].direction = direction

        return True

    @ti.func
    def reflectance(self, cosine, ref_idx):
        r0 = (1 - ref_idx) / (1 + ref_idx)
        r0_new = r0 * r0
        return r0_new + (1-r0_new)*ti.pow((1- cosine), 5)


# Hit record
class HitRecord:
    def __init__(self, res):
        self.record = ti.Struct.field({
            "p": ti.types.vector(3, ti.f32),
            "normal": ti.types.vector(3, ti.f32),
            "t": ti.f32,
            "front_face": ti.f32,
            "material_index": ti.i32,
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
        center=vec3f, raduis=ti.f32, material_index=ti.i32
        )

        self.size = len(descriptor)
        self.content = shpere_field.field(shape=(self.size,))

        for i, obj in enumerate(descriptor):
            self.content[i].center = ti.Vector(obj["center"])
            self.content[i].raduis = obj["raduis"]
            self.content[i].material_index = obj["material"]

    
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

            hit_holder.record[i, j].material_index = self.content[index].material_index


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
        for index in ti.static(range(shpere.size)):
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
    def __init__(self, lookfrom, lookat, vup, vfov, aspect_ratio, aperture, focus_dist, image_width):

        theta = np.deg2rad(vfov)
        h = np.tan(theta/2)
        viewport_height = 2.0 * h
        viewport_width = aspect_ratio * viewport_height

        w = unit_vector(lookfrom - lookat)
        u = unit_vector(vup.cross(w))
        v = w.cross(u)

        image_height = int(image_width / aspect_ratio) 


        self.viewport_height = ti.field(float, shape = ())
        self.viewport_width = ti.field(float, shape = ())
        self.horizontal = ti.Vector.field(3, float, shape = ())
        self.vertical = ti.Vector.field(3, float, shape = ())
        self.origin = ti.Vector.field(3, float, shape = ())
        self.lower_left_corner = ti.Vector.field(3, float, shape = ())
        self.focal_length = ti.field(float, shape = ())
        self.lens_radius = ti.field(float, shape = ())
        self.focus_dist = ti.field(float, shape = ())

        self.w = ti.Vector.field(3, float, shape = ())
        self.u = ti.Vector.field(3, float, shape = ())
        self.v = ti.Vector.field(3, float, shape = ())

        self.lookfrom = lookfrom
        self.lookat = lookat
        self.aspect_ratio = aspect_ratio
        self.vup = vup

        self.w[None] = w
        self.u[None] = u
        self.v[None] = v

        self.focus_dist[None] = focus_dist

        origin = lookfrom
        horizontal = ti.Vector([u[0], u[1], u[2]]) * viewport_width * focus_dist
        vertical = ti.Vector([v[0], v[1], v[2]]) * viewport_height * focus_dist

        self.origin[None] = origin
        self.horizontal[None] = horizontal
        self.vertical[None] = vertical
        self.lower_left_corner[None] = origin - horizontal/2 - vertical/2 - w * focus_dist

        self.lens_radius[None] = aperture / 2

        self.res = (image_width, image_height)
        self.samples_per_pixel = 10

    def update_camera(self, local_movement, oriented_movement, vfov, focus_dist):

        w = field1D_to_vector(self.w)
        u = field1D_to_vector(self.u)
        v = field1D_to_vector(self.v)

        lookfrom_vector = self.lookfrom - local_movement[1] * w + local_movement[0] * u + oriented_movement * u
        lookat_vector = self.lookat + local_movement[0] * u 

        theta = degrees_to_radians(vfov)
        h = ti.tan(theta/2)
        viewport_height = 2.0 * h
        viewport_width = self.aspect_ratio * viewport_height

        w = unit_vector(lookfrom_vector - lookat_vector)
        u = unit_vector(self.vup.cross(w))
        v = w.cross(u)

        self.w[None] = w
        self.u[None] = u
        self.v[None] = v

        origin = lookfrom_vector
        horizontal = ti.Vector([u[0], u[1], u[2]]) * viewport_width * self.focus_dist[None]
        vertical = ti.Vector([v[0], v[1], v[2]]) * viewport_height * self.focus_dist[None]
        lower_left_corner = origin - horizontal/2 - vertical/2 - w * self.focus_dist[None]

        return origin, horizontal, vertical, lower_left_corner

    @ti.func
    def set_ray(self, i, j, s, t):

        rd = self.lens_radius[None] * random_in_unit_disk()
        offset = self.u[None] * rd[0] + self.v[None] * rd[1]

        ray_holder.rays[i, j].origin = self.origin[None] + offset
        ray_holder.rays[i, j].direction = self.lower_left_corner[None] + s * self.horizontal[None] + t * self.vertical[None] - self.origin[None] - offset


max_depth = 50

vfov = 20.0
R = np.cos(pi/4)
aspect_ratio = 16.0 / 9.0
image_width = 1280
lookfrom_list = [3.0,3.0,2.0]
lookat_list = [0.0,0.0,-1.0]
aperture = 0.1
temp_lf_la = ti.Vector(lookfrom_list) - ti.Vector(lookat_list)
dist_to_focus = ti.sqrt(temp_lf_la.dot(temp_lf_la))
camera = Camera(
    ti.Vector(lookfrom_list), 
    ti.Vector(lookat_list), 
    ti.Vector([0.0,1.0,0.0]), 
    vfov, 
    aspect_ratio,
    aperture, 
    dist_to_focus,
    image_width
)

# material
material_ground = Lambertian(ti.Vector([0.8, 0.8, 0.0]))
# material_center = Lambertian(ti.Vector([0.7, 0.3, 0.3]))
# material_center = Dielectric(1.5)
material_center = Lambertian(ti.Vector([0.1, 0.2, 0.5]))
# material_left = Metal(ti.Vector([0.8, 0.8, 0.8]), 0.3)
material_left = Dielectric(1.5)
# material_right = Metal(ti.Vector([0.8, 0.6, 0.2]), 1.0)
material_right = Metal(ti.Vector([0.8, 0.6, 0.2]), 0.0)
ground_index = 0
center_index = 1
left_index = 2
right_index = 3

# hit_holder = HitRecord(camera.res)
hit_holder = HitRecord(camera.res)
ray_holder = Ray(camera.res)

world = HittableList()

shpere_descriptor = []
shpere_descriptor.append({"shape_name": "shpere", "center": [ 0.0, -100.5, -1.0], "raduis":  100, "material": ground_index})
shpere_descriptor.append({"shape_name": "shpere", "center": [ 0.0,    0.0, -1.0], "raduis":  0.5, "material": center_index})
shpere_descriptor.append({"shape_name": "shpere", "center": [-1.0,    0.0, -1.0], "raduis":  0.5, "material": left_index})
shpere_descriptor.append({"shape_name": "shpere", "center": [-1.0,    0.0, -1.0], "raduis":-0.45, "material": left_index})
shpere_descriptor.append({"shape_name": "shpere", "center": [ 1.0,    0.0, -1.0], "raduis":  0.5, "material": right_index})


shpere = Shpere(shpere_descriptor)
world.add(shpere_descriptor)

image_width = camera.res[0]
image_height = camera.res[1]
samples_per_pixel = camera.samples_per_pixel

# Render
radience = ti.Vector.field(3, dtype = float, shape=camera.res)
final_pixels = ti.Vector.field(3, dtype=float, shape=camera.res)

@ti.kernel
def reset_kernel(lookfrom_vector: ti.template(), horizontal: ti.template(), vertical: ti.template(), lower_left_corner: ti.template()):

    camera.origin[None] = lookfrom_vector
    camera.horizontal[None] = horizontal
    camera.vertical[None] = vertical
    camera.lower_left_corner[None] = lower_left_corner

@ti.kernel
def render(cnt : int):
    for i, j in ti.ndrange((0, image_width), (0, image_height)):

        pixel_color = ti.Vector([0.0, 0.0, 0.0])
        for s in range(samples_per_pixel):
            u = (i + ti.random(ti.f32)) / (image_width - 1)
            v = (j + ti.random(ti.f32)) / (image_height-1)

            # ray
            camera.set_ray(i, j, u, v)

            color = ti.Vector([0.0,0.0,0.0])
            cur_attenuation = ti.Vector([1.0, 1.0, 1.0])
            for d in range(max_depth):
                hitting = world.hit(i, j, 0.001, infinity)
                if hitting:

                    attenuation = ti.Vector([1.0, 1.0, 1.0])
                    did_scatter = False
                    if hit_holder.record[i, j].material_index == 0:
                        did_scatter = material_ground.scatter(i, j, attenuation)
                    elif hit_holder.record[i, j].material_index == 1:
                        did_scatter = material_center.scatter(i, j, attenuation)
                    elif hit_holder.record[i, j].material_index == 2:
                        did_scatter = material_left.scatter(i, j, attenuation)
                    elif hit_holder.record[i, j].material_index == 3:
                        did_scatter = material_right.scatter(i, j, attenuation)
                    
                    if did_scatter:
                        cur_attenuation *= attenuation
                    else:
                        break
                else:
                    unit_direction = unit_vector(ray_holder.rays[i, j].direction)
                    t = 0.5*(unit_direction[1] + 1.0)
                    color = ((1.0-t)*ti.Vector([1.0, 1.0, 1.0]) + t*ti.Vector([0.5, 0.7, 1.0])) * cur_attenuation
                    break

            pixel_color += color
        write_color(final_pixels, i, j, pixel_color, samples_per_pixel, cnt)


window = ti.ui.Window("Taichi RayTracer", camera.res, vsync=True)
canvas = window.get_canvas()
pause = False
local_movement = [0.0, 0.0, 0.0]
movement_speed = 0.05
oriented_movement = 0.0
cnt = 0
while window.running:
    if window.get_event(ti.ui.PRESS):
        if window.event.key in [ti.ui.ESCAPE]:
            exit()
        if window.event.key == 'p':
            pause = not pause
            print(pause)
        if window.event.key in ['a', 'w', 's', 'd', 'q', 'e', 'z', 'c']:
            if window.event.key == 'a':
                local_movement[0] = local_movement[0] - movement_speed
            if window.event.key == 'w':
                local_movement[1] = local_movement[1] + movement_speed
            if window.event.key == 's':
                local_movement[1] = local_movement[1] - movement_speed
            if window.event.key == 'd':
                local_movement[0] = local_movement[0] + movement_speed

            if window.event.key == 'q':
                vfov += 5
            if window.event.key == 'e':
                vfov -= 5

            if window.event.key == 'z':
                oriented_movement -= movement_speed
            if window.event.key == 'c':
                oriented_movement += movement_speed

            # print(window.event.key)

            cnt = 0
            radience.fill(0)

            origin, horizontal, vertical, lower_left_corner = camera.update_camera(ti.Vector(local_movement), oriented_movement, vfov)
            reset_kernel(origin, horizontal, vertical, lower_left_corner)
            print(camera.origin)


    cnt += 1
    if pause == False:
        render(cnt)
    # print(final_pixels)
    canvas.set_image(final_pixels)
    window.show()

