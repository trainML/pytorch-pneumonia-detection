import { useState } from 'react';
import axios from 'axios';
import Grid from '@material-ui/core/Grid';
import Container from '@material-ui/core/Container';

import config from './config';
import Input from './Input';
import Results from './Results';
import Header from './Header';

const uploadImage = async ({ pId, dicom }) => {
  if (dicom) {
    try {
      const resp = await axios.post(
        `${config.api_address}${config.route_path}`,
        {
          pId,
          dicom,
        }
      );
      return resp.data;
    } catch (error) {
      if (error.response) {
        console.log(error.response.status);
        console.log(error.response);
      } else {
        console.log(error);
      }
    }
  }
};

function App() {
  const [input, setInput] = useState({});
  const [predictions, setPredictions] = useState([]);

  return (
    <Grid>
      <Header />
      <br />
      <Container>
        <Grid container spacing={2}>
          <Grid item xs={6}>
            <Input
              onChange={setInput}
              onSubmit={async () => {
                const result = await uploadImage(input);
                if (result) {
                  const [pId, prediction] = Object.entries(result)[0];
                  setPredictions((predictions) => {
                    const new_predictions = [...predictions];
                    new_predictions.push({ pId, prediction });
                    return new_predictions;
                  });
                }
              }}
            />
          </Grid>
          <Grid item xs={6}>
            {predictions ? <Results predictions={predictions} /> : undefined}
          </Grid>
        </Grid>
      </Container>
    </Grid>
  );
}

export default App;
