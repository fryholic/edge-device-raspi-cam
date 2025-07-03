#include <iostream>
#include <cstring>
#include <mqtt/async_client.h>

const std::string SERVER_ADDRESS("tcp://localhost:1883");
const std::string CLIENT_ID("subscriber");
const std::string TOPIC("test/topic");
const int QOS = 1;

class callback : public virtual mqtt::callback {
public:
    void message_arrived(mqtt::const_message_ptr msg) override {
        std::cout << "Message received: "
                  << msg->get_topic() << ": " 
                  << msg->to_string() << std::endl;
    }
};

int main() {
    mqtt::async_client client(SERVER_ADDRESS, CLIENT_ID);
    callback cb;
    client.set_callback(cb);

    mqtt::connect_options connOpts;
    connOpts.set_clean_session(true);

    try {
        client.connect(connOpts)->wait();
        client.subscribe(TOPIC, QOS)->wait();
        std::cout << "Subscribed to topic: " << TOPIC << std::endl;

        // 메시지 수신 대기 (Ctrl+C로 종료)
        while (true) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
    catch (const mqtt::exception& exc) {
        std::cerr << "Error: " << exc.what() << std::endl;
        return 1;
    }
    return 0;
}