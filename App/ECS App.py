from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
import socket

class QuadcopterApp(App):

    def build(self):
        self.layout = BoxLayout(orientation='vertical')

        self.takeoff_button = Button(text='Takeoff', on_press=self.takeoff)
        self.layout.add_widget(self.takeoff_button)

        self.land_button = Button(text='Land', on_press=self.land)
        self.layout.add_widget(self.land_button)

        # Add buttons for flying directions
        self.forward_button = Button(text='Forward', on_press=self.fly_forward)
        self.layout.add_widget(self.forward_button)

        self.backward_button = Button(text='Backward', on_press=self.fly_backward)
        self.layout.add_widget(self.backward_button)

        self.left_button = Button(text='Left', on_press=self.fly_left)
        self.layout.add_widget(self.left_button)

        self.right_button = Button(text='Right', on_press=self.fly_right)
        self.layout.add_widget(self.right_button)

        return self.layout

    def takeoff(self, instance):
        self.send_command('takeoff')

    def land(self, instance):
        self.send_command('land')

    def fly_forward(self, instance):
        self.send_command('forward')

    def fly_backward(self, instance):
        self.send_command('backward')

    def fly_left(self, instance):
        self.send_command('left')

    def fly_right(self, instance):
        self.send_command('right')

    def send_command(self, command):
        # Raspberry Pi IP and port
        pi_ip = '192.168.1.100'  # Replace with your Raspberry Pi IP
        pi_port = 5555

        client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            client_socket.sendto(command.encode('utf-8'), (pi_ip, pi_port))
        finally:
            client_socket.close()

if _name_ == '_main_':
    QuadcopterApp().run()
