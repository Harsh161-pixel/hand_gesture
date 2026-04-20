from pythonosc import dispatcher, osc_server
import bpy
import threading

current_gesture = "No Hand Detected"
current_mode = "object"

def gesture_handler(address, *args):
    global current_gesture
    if args:
        current_gesture = str(args[0])

def mode_handler(address, *args):
    global current_mode
    if args:
        current_mode = str(args[0])

disp = dispatcher.Dispatcher()
disp.map("/gesture", gesture_handler)
disp.map("/mode", mode_handler)

server = osc_server.ThreadingOSCUDPServer(("127.0.0.1", 9000), disp)
print("OSC Server started on port 9000 - waiting for gesture.")

server_thread = threading.Thread(target=server.serve_forever, daemon=True)
server_thread.start()

def control_loop():
    global current_gesture,current_mode

    obj = bpy.context.active_object

    if not obj:
        return 0.08


    if current_mode == "object":
        if current_gesture == "open palm":
            obj.location.x += 0.08
        elif current_gesture == "pinch":
            obj.scale *= 1.03
        elif current_gesture == "fist":
            obj.location.x -= 0.08
        elif current_gesture == "pointing":
            obj.rotation_euler.z += 0.1
        elif current_gesture == "thumbs up":
            try:
                bpy.ops.mesh.extrude_region_move(TRANSFORM_OT_translate={"value": (0,0,0.2)})
            except:
                pass

    else:
        if current_gesture == "open palm":
            bpy.ops.view3d.rotate(mode='ORBIT', value=0.05)
        elif current_gesture == "pinch":
            bpy.ops.view3d.zoom(delta=10)
        elif current_gesture == "fist":
            bpy.ops.view3d.pan(direction='LEFT', value=0.1)

    return 0.05

bpy.app.timers.register(control_loop)
print("blender OSC Receiver Ready!")
