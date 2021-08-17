import { useState } from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Box from '@material-ui/core/Box';
import Grid from '@material-ui/core/Grid';
import Typography from '@material-ui/core/Typography';
import Paper from '@material-ui/core/Paper';
import Button from '@material-ui/core/Button';
import grey from '@material-ui/core/colors/grey';
import SendIcon from '@material-ui/icons/Send';
import UploadButton from './UploadButton';

const useStyles = makeStyles((theme) => ({
  image: {
    maxWidth: '100%',
    maxHeight: '100%',
  },
  paper: {
    height: '600px',
    alignContent: 'center',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: grey[300],
  },
}));

const getDicom = async (e) => {
  const pId = e.target.files[0].name.replace('.dcm', '');
  const file = await new Promise((resolve, reject) => {
    var fr = new FileReader();
    fr.onload = () => {
      resolve(fr.result);
    };
    fr.readAsDataURL(e.target.files[0]);
  });
  const base64result = file.split(',')[1];

  return { pId, dicom: base64result };
};

function Input({ onChange, onSubmit }) {
  const classes = useStyles();
  const [pId, setpId] = useState();
  const [submitting, setSubmitting] = useState(false);

  if (pId) {
    return (
      <Grid container direction='column' xs={12}>
        <Grid item xs={12}>
          <Box textAlign='center'>
            <Typography variant='h5'>Input Patient ID</Typography>

            <Typography variant='subtitle1'>{pId}</Typography>
          </Box>
        </Grid>
        <br />
        <Grid item xs={12}>
          <Box textAlign='center'>
            <Grid container justifyContent='center' xs={12} spacing={3}>
              <Grid item>
                <UploadButton
                  name='Upload New File'
                  onChange={async (e) => {
                    const dicom = await getDicom(e);
                    setpId(dicom.pId);
                    onChange(dicom);
                  }}
                />
              </Grid>
              <Grid item>
                <Button
                  variant='contained'
                  color='primary'
                  icon='send'
                  disabled={submitting}
                  endIcon={<SendIcon />}
                  onClick={async (e) => {
                    e.preventDefault();
                    setSubmitting(true);
                    await onSubmit();
                    setSubmitting(false);
                  }}
                >
                  Get Prediction
                </Button>
              </Grid>
            </Grid>
          </Box>
        </Grid>
      </Grid>
    );
  } else {
    return (
      <Paper variant='outlined' className={classes.paper}>
        <UploadButton
          name='Upload File'
          onChange={async (e) => {
            const dicom = await getDicom(e);
            setpId(dicom.pId);
            onChange(dicom);
          }}
        />
      </Paper>
    );
  }
}

export default Input;
