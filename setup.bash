#!/bin/bash

# Part 2: Configuration variables
PROJECT_ID="beta-deployment-382610"
ZONE="us-central1-a"
MACHINE_TYPE="e2-highmem-16"
IMAGE_PROJECT="debian-cloud"
DISK_IMAGE="debian-12-bookworm-v20250415"
DISK_SIZE="110GB"
NETWORK_TAG="chromadb-server"

# Part 3: VM name sequence from part2 to part8
for i in {2..8}; do
  INSTANCE_NAME="chromadb-props-part${i}"
  echo "Creating VM instance: $INSTANCE_NAME..."

  # Part 4: Create VM
  gcloud compute instances create "$INSTANCE_NAME" \
    --project="$PROJECT_ID" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --boot-disk-size="$DISK_SIZE" \
    --boot-disk-type="pd-balanced" \
    --boot-disk-device-name="$INSTANCE_NAME" \
    --image="$DISK_IMAGE" \
    --image-project="$IMAGE_PROJECT" \
    --tags="$NETWORK_TAG" \
    --quiet

  # Part 5: Confirm tag assignment
  echo "Assigned network tag: $NETWORK_TAG to $INSTANCE_NAME"
done

# Part 6: Completion message
echo "==============================================="
echo "✅ All chromadb-props-part2 to part8 VMs created successfully."
echo "➡️  Machine type: $MACHINE_TYPE"
echo "➡️  Disk image: $DISK_IMAGE ($DISK_SIZE)"
echo "➡️  Network tag: $NETWORK_TAG"
echo "➡️  Project: $PROJECT_ID | Zone: $ZONE"
echo "==============================================="

# Part 7: Exit
exit 0