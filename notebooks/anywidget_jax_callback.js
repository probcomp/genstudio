import {createRender, useModel, useExperimental} from "https://esm.sh/@anywidget/react@0.0.7"
import * as React from 'https://esm.sh/react@18.3.1'
import htm from 'https://esm.sh/htm@3.1.1'

const {useState, useEffect} = React
const html = htm.bind(React.createElement)

  const useCustomMessages = (model) => {
      const [messages, setMessages] = useState([])
      const handleMessage = (message, info) => {
        if (message.kind !== 'genstudio') {
          return;
        }
        setMessages((ms) => [...ms, message.content]);
      };
      useEffect(() => {
        model.on("msg:custom", handleMessage)
        return () => model.off("msg:custom", handleMessage)
      }, [model])
      return messages
  }

  const App = () => {
    const model = useModel()
    const {invoke} = useExperimental()
    const messages = useCustomMessages(model)
        useEffect(() => {
            if (messages.length > 0) {
                // Send a response to python
                invoke("_receive_message", messages[messages.length - 1], [])
            }
            },
        [messages.join()])
        return html`<div>
                    Messages: ${messages.join(', ')}
                </div>`
    }
  export default {render: createRender(App)}
