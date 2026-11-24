# habit_combine.py HABIT.blend MINE.blend OUT.blend : the habit (blender/base/views.py --figure habit_all --mannequin --turn --save HABIT.blend)
# with the lapel-and-collar coat of a notchcollar build (MINE.blend: Redingote bodice, buttons, collar) in place of its plain bodice
import bpy, sys
habit, mine, out = sys.argv[-3:]
bpy.ops.wm.open_mainfile(filepath=habit)
before = set(bpy.data.objects.keys()); mats0 = set(bpy.data.materials.keys())
old = bpy.data.objects['Redingote bodice']; oldb = bpy.data.objects['Redingote bodice buttons']
wool = old.material_slots[0].material if old.material_slots else None
print('combine: habit wool material:', wool.name if wool else None, 'parent', old.parent.name if old.parent else None)
names = ['Redingote bodice', 'Redingote bodice buttons', 'Redingote collar']
with bpy.data.libraries.load(mine, link=False) as (src, dst):
    dst.objects = [n for n in names if n in src.objects]
new = set(bpy.data.objects.keys()) - before
print('combine: appended', sorted(new))
rig = bpy.data.objects['Rig']
get = lambda n: bpy.data.objects.get(n)
sc = bpy.context.scene
for n in sorted(new):
    o = bpy.data.objects[n]
    if o.users_collection == () or not o.users_collection: sc.collection.objects.link(o)
old.name = 'Redingote bodice (habit)'; oldb.name = 'Redingote bodice buttons (habit)'
for o in (old, oldb): o.hide_render = True; o.hide_viewport = True
for n in sorted(new):
    o = bpy.data.objects[n]
    if o.name.endswith('.001'): o.name = o.name[:-4]
    for m in o.modifiers:
        if m.type == 'ARMATURE': m.object = rig
    if o.parent is not None:
        pn = o.parent.name[:-4] if o.parent.name.endswith('.001') else o.parent.name
        o.parent = bpy.data.objects.get(pn)
    print('combine:', o.name, 'parent', o.parent.name if o.parent else None, [m.type for m in o.modifiers])
# the extra copies pulled in as dependencies
for n in list(bpy.data.objects.keys()):
    if n not in before and bpy.data.objects[n].name not in ('Redingote bodice', 'Redingote bodice buttons', 'Redingote collar'):
        print('combine: removing dependency copy', n); bpy.data.objects.remove(bpy.data.objects[n], do_unlink=True)
if wool:
    for n in ('Redingote bodice', 'Redingote collar'):
        o = bpy.data.objects[n]; o.data.materials.clear(); o.data.materials.append(wool)
bpy.ops.wm.save_as_mainfile(filepath=out)
