# Automated Azure setup

This guide creates the same resources as
[`manual_setup.md`](manual_setup.md) using the Azure CLI, which is the version to
use in CI or when you would rather not click through the portal.

You will end up with a resource group, a container registry and a web app, and
a service principal that lets GitHub Actions deploy to it.

## What you need

- An Azure subscription
- The Azure CLI
- Docker, to push the first image

Set these once and reuse them in every command below:

```bash
export AZURE_SUBSCRIPTION_ID="00000000-0000-0000-0000-000000000000"
export AZURE_LOCATION="eastus"
export APP_NAME="my-segmentation-app"   # must be globally unique
```

Derive the resource names from the app name:

| Setting | Value |
|---------|-------|
| Resource group | `${APP_NAME}rg` |
| Registry | `${APP_NAME}acr` |
| Web app | `${APP_NAME}-web` |

## 1. Install the Azure CLI

**Windows** (PowerShell, elevated):

```powershell
$ProgressPreference = 'SilentlyContinue'
Invoke-WebRequest -Uri https://aka.ms/installazurecliwindows -OutFile .\AzureCLI.msi
Start-Process msiexec.exe -Wait -ArgumentList '/I AzureCLI.msi /quiet'
Remove-Item .\AzureCLI.msi
```

**macOS**:

```bash
brew install azure-cli
```

**Debian / Ubuntu**:

```bash
curl -L https://aka.ms/InstallAzureCli | bash
```

**Red Hat / CentOS / Fedora**:

```bash
sudo dnf install -y azure-cli
```

Confirm the install:

```bash
az version
```

## 2. Sign in

For interactive use:

```bash
az login
```

Your browser opens, you approve the request, and you are signed in. Verify which
subscription is active with `az account show`.

In CI there is no browser, so use a service principal instead - see
[step 6](#6-create-the-service-principal-for-github-actions).

## 3. Create the resources

Everything below is also available as a script:

```bash
bash scripts/create_azure_resources.sh
```

To run the commands by hand:

```bash
# Resource group
az group create \
    --name "${APP_NAME}rg" \
    --location "$AZURE_LOCATION"

# Container registry
az acr create \
    --resource-group "${APP_NAME}rg" \
    --name "${APP_NAME}acr" \
    --sku Basic \
    --location "$AZURE_LOCATION" \
    --admin-enabled true

# Web app running a container from that registry
az webapp create \
    --resource-group "${APP_NAME}rg" \
    --name "${APP_NAME}-web" \
    --container-image-name "${APP_NAME}acr.azurecr.io/customer-segmentation:latest" \
    --container-registry-server "${APP_NAME}acr.azurecr.io" \
    --container-registry-username "$(az acr credential show \
        --name "${APP_NAME}acr" --query username --output tsv)" \
    --container-registry-password "$(az acr credential show \
        --name "${APP_NAME}acr" --query passwords[0].value --output tsv)" \
    --runtime-version "1.0" \
    --https-only true

# Tell the platform which port the container listens on
az webapp config appsettings set \
    --resource-group "${APP_NAME}rg" \
    --name "${APP_NAME}-web" \
    --settings \
        WEBSITES_PORT=5000 \
        WEBSITES_CONTAINER_START_TIME_LIMIT=60 \
        APP_PORT=5000
```

> The image has to exist in the registry before the web app is created. Push it
> first, following the next step, or run `az webapp create` with a placeholder
> and update the image afterwards with `az webapp deploy`.

## 4. Push the application image

```bash
export AZURE_LOGIN_SERVER="${APP_NAME}acr.azurecr.io"

docker login "$AZURE_LOGIN_SERVER" \
    --username "$(az acr credential show --name "${APP_NAME}acr" --query username --output tsv)" \
    --password "$(az acr credential show --name "${APP_NAME}acr" --query passwords[0].value --output tsv)"

docker build -t "$AZURE_LOGIN_SERVER/customer-segmentation:latest" .
docker push "$AZURE_LOGIN_SERVER/customer-segmentation:latest"
```

## 5. Add the application settings

The secrets the app needs are set here rather than baked into the image:

```bash
az webapp config appsettings set \
    --resource-group "${APP_NAME}rg" \
    --name "${APP_NAME}-web" \
    --settings \
        MONGO_DB_URL="$MONGO_DB_URL" \
        AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
        AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
        AWS_DEFAULT_REGION="ap-south-1" \
        CORS_ORIGINS="https://${APP_NAME}-web.azurewebsites.net"
```

Verify:

```bash
curl "https://${APP_NAME}-web.azurewebsites.net/health"
# {"status":"ok"}
```

## 6. Create the service principal for GitHub Actions

The deploy workflow authenticates with a service principal instead of a user
account. Scope it to the resource group so it cannot touch anything else in
your subscription:

```bash
az ad sp create-for-rbac \
    --name "${APP_NAME}-deploy" \
    --role contributor \
    --scopes "/subscriptions/${AZURE_SUBSCRIPTION_ID}/resourceGroups/${APP_NAME}rg" \
    --sdk-auth
```

The JSON output contains everything the workflow needs. Store it as a single
repository secret called `AZURE_CREDENTIALS` - see
[`deployment.md`](deployment.md). The other values (`AZURE_LOGIN_SERVER`,
`REGISTRY_USERNAME`, `REGISTRY_PASSWORD`, `REPO_NAME`, `AZURE_WEB_APP_NAME`)
come from the commands above.

Grant the registry pull role if the workflow needs to authenticate separately:

```bash
ACR_ID=$(az acr show --name "${APP_NAME}acr" --query id --output tsv)
SP_ID=$(az ad sp list --filter "displayName eq '${APP_NAME}-deploy" --query "[0].id" --output tsv)

az role assignment create \
    --role acrpull \
    --assignee "$SP_ID" \
    --scope "$ACR_ID"
```

## 7. Clean up

```bash
az group delete --resource-group "${APP_NAME}rg" --yes
```

Or:

```bash
bash scripts/delete_azure_resources.sh
```

Deleting the resource group removes the registry, the web app and their
contents, so you stop being billed for them.
