FROM condaforge/mambaforge

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get upgrade -y && apt install gcc -y && apt install build-essential -y

# Install environment
COPY environment.yml ./
RUN mamba env create -f environment.yml -n websmsreg && \
    mamba clean -a

# Set mrsa-workflow environment as active at start-up
RUN echo "source activate websmsreg" >> ~/.bashrc

# Add environment to PATH
ENV PATH /opt/conda/envs/websmsreg/bin:${PATH}


COPY ./app.py /code/
RUN mkdir /code/assets
COPY ./assets /code/assets

WORKDIR /code/

# Open up port 8888
EXPOSE 8000

CMD ["python3", "app.py"]
