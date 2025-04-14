FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime 

ENV prefetch_factor=None

RUN apt update; exit 0
RUN apt install -y nano
RUN apt-get update && apt-get install -y libgl1
RUN apt-get update && apt-get install -y libglib2.0-0
RUN apt-get update && apt-get install -y git  

# create app directory and user
RUN mkdir /app
RUN chmod a+rw /app

#replace 1001/1007 with your user/group id 
# get uid: id -u
# get gid: id -g 
RUN groupadd -g 1001 app && useradd -r -u 1007 -g app -d /app -s /sbin/nologin -c "non-root app user" app

WORKDIR /workspace
RUN chown -R app:app /workspace
RUN chmod -R a+rw /workspace


COPY --chown=app:app requirements.txt /workspace

RUN python -m pip install --upgrade pip && pip install --root-user-action=ignore -r requirements.txt
RUN pip install gymnasium
RUN pip install -U ipykernel
# RUN jupyter nbextension enable --py widgetsnbextension
# RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager
# COPY --chown=app:app routines /workspace
# RUN mkdir /workspace/output_files

USER app

