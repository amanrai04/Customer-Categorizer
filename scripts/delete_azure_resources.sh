#!/usr/bin/env bash
#
# Delete every Azure resource created by create_azure_resources.sh.
#
# Deleting the resource group removes the container registry, the web app and
# their contents, so nothing keeps being billed after a demo.
#
# Environment variables:
#   APP_NAME                the prefix used to create the resources
#
# Optional:
#   CONFIRM_DESTROY         set to anything other than "yes" for a dry run

set -euo pipefail

: "${APP_NAME:?APP_NAME is required}"

RESOURCE_GROUP="${APP_NAME}rg"

if ! az account show >/dev/null 2>&1; then
    echo "Not signed in to Azure. Run 'az login' first." >&2
    exit 1
fi

if ! az group show --name "$RESOURCE_GROUP" >/dev/null 2>&1; then
    echo "Resource group ${RESOURCE_GROUP} does not exist, nothing to delete."
    exit 0
fi

if [[ "${CONFIRM_DESTROY:-no}" != "yes" ]]; then
    echo "About to delete resource group: ${RESOURCE_GROUP}"
    echo "Re-run with CONFIRM_DESTROY=yes to delete it."
    exit 0
fi

az group delete --resource-group "$RESOURCE_GROUP" --yes

echo "Deleted ${RESOURCE_GROUP}."
