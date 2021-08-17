import { makeStyles } from '@material-ui/core/styles';
import Box from '@material-ui/core/Box';
import Grid from '@material-ui/core/Grid';

import Paper from '@material-ui/core/Paper';
import Typography from '@material-ui/core/Typography';

const useStyles = makeStyles((theme) => ({
  paper: {
    backgroundColor: theme.palette.primary.main,
    color: '#fff',
  },
  image: {
    maxWidth: '100%',
    maxHeight: '100%',
  },
  annotations: {
    backgroundColor: theme.palette.secondary.main,
  },
}));

function Results({ predictions }) {
  const classes = useStyles();
  if (!predictions.length) {
    return null;
  } else {
    const reversed = [...predictions].reverse();
    return (
      <>
        <Box textAlign='center'>
          <Typography variant='h5'>Predictions</Typography>
        </Box>
        {reversed.map((prediction, i) => {
          return (
            <Grid container direction='column' key={i}>
              <br />
              <Paper variant='outlined' className={classes.paper}>
                <Grid item xs={12}>
                  <Box textAlign='center'>
                    <Typography variant='subtitle1'>
                      {prediction.pId}
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={12}>
                  <Box textAlign='center' className={classes.annotations}>
                    <Typography variant='subtitle1'>
                      <b>Boxes: </b>
                      {prediction.prediction.annotations || 'None'}
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={12}>
                  <Box textAlign='center'>
                    <img
                      className={classes.image}
                      alt='uploaded'
                      src={`data:image/png;base64,${prediction.prediction.image}`}
                    />
                  </Box>
                </Grid>
              </Paper>
            </Grid>
          );
        })}
      </>
    );
  }
}

export default Results;
