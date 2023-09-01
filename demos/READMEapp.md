# Water Level Classification for Bottles: App

Click [here](https://github.com/emirhansilsupur/water-bottle-image-classification/blob/09f643a181e47fc82c80dbc6d565ba19790d0796/README.md) for information about the project. 

## Prerequisites

Create venv before install requirements

python = 3.10.x
```
pip install -r requirements.txt
```
## Installation

Open a terminal or command prompt on your computer.
```
docker pull emirhnslspr/wbottleclf:v1.0
```
This command instructs Docker to pull an image named wbottleclf from the [Docker Hub](https://hub.docker.com/r/emirhnslspr/wbottleclf) registry.

## Usage

To run the Docker image you've pulled, use the following command:
```
docker container run --name container_name -p 8000:8000 emirhnslspr/wbottleclf:v1.0
```
- **--name :** Assign a custom name to your container. Replace container_name with a name of your choice.
- **-p :** Map port 8000 from your local machine to port 8000 inside the container.
- **emirhnslspr/wbottleclf:v1.0 :** Specify the image to use for creating the container.

**Access the Application:** With the container running, you can access your application by opening a web browser and navigating to http://localhost:8000 (or the appropriate URL based on your application's configuration). Your Dockerized application should now be up and running.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)