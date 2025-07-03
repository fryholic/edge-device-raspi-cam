#include <iostream>
#include <cstring>
#include <thread>
#include <mqtt/async_client.h>

const std::string SERVER_ADDRESS("tcp://localhost:1883");
const std::string CLIENT_ID("publisher");
const std::string TOPIC("test/topic");
const int QOS = 1;

int main() {
    mqtt::async_client client(SERVER_ADDRESS, CLIENT_ID);
    mqtt::connect_options connOpts;
    connOpts.set_clean_session(true);

    try {
        client.connect(connOpts)->wait();
        std::cout << "Connected to broker!" << std::endl;

        int msg_count = 0;
        while (true) {
            std::string payload = "Message " + std::to_string(++msg_count);
            auto pubmsg = mqtt::make_message(TOPIC, payload, QOS, false);
            client.publish(pubmsg)->wait();
            std::cout << "Published: " << payload << std::endl;

            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
    catch (const mqtt::exception& exc) {
        std::cerr << "Error: " << exc.what() << std::endl;
        return 1;
    }
    return 0;
}