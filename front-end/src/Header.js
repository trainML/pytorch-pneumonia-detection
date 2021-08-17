import { makeStyles } from '@material-ui/core/styles';
import Box from '@material-ui/core/Box';
import Grid from '@material-ui/core/Grid';
import Typography from '@material-ui/core/Typography';
import Paper from '@material-ui/core/Paper';

const useStyles = makeStyles((theme) => ({
  logo: {
    width: '250px',
  },
  paper: {
    padding: theme.spacing(1),
    textAlign: 'center',
    backgroundColor: theme.palette.brandBlack.main,
  },
  header: {
    color: theme.palette.brandBlack.contrastText,
    paddingTop: '15px',
  },
}));

function Header() {
  const classes = useStyles();
  return (
    <Paper className={classes.paper}>
      <Grid container xs={12}>
        <Grid item xs={12} md={3}>
          <a href='https://www.trainml.ai'>
            <img
              alt='logo'
              className={classes.logo}
              src='https://www.trainml.ai/static/img/trainML-logo-purple.png'
            />
          </a>
        </Grid>
        <Grid item xs={12} md={9}>
          <Box textAlign='left'>
            <Typography variant='h3' className={classes.header}>
              Pneumonia Detection Endpoint
            </Typography>
          </Box>
        </Grid>
      </Grid>
    </Paper>
  );
}

export default Header;
