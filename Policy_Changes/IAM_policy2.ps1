# Define variables
$BucketName = "gs://med-labs-42f13"
$Member = "allUsers"
$Role = "roles/storage.objectViewer"

# Authenticate gcloud if not already authenticated
if (-not (gcloud auth list --filter=status:ACTIVE 2>$null)) {
    Write-Host "Authenticating gcloud..."
    gcloud auth application-default login
}

# Disable public access prevention on the bucket
Write-Host "Attempting to disable public access prevention on $BucketName..."
gcloud storage buckets update $BucketName --no-public-access-prevention

# Check the exit code for the update command
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to disable public access prevention. This might be due to an organization policy. Check with your admin."
    # Attempt to describe the bucket to confirm PAP status
    $papStatus = gcloud storage buckets describe $BucketName --format="value(iamConfiguration.publicAccessPrevention)"
    Write-Host "Current PAP status: $papStatus"
    exit 1
} else {
    Write-Host "Public access prevention disabled successfully."
}

# Add a delay to allow changes to propagate
Start-Sleep -Seconds 30

# Verify PAP status again
$papStatus = gcloud storage buckets describe $BucketName --format="value(iamConfiguration.publicAccessPrevention)"
if ($papStatus -eq "enforced") {
    Write-Host "Public access prevention is still enforced after delay. Exiting."
    exit 1
} else {
    Write-Host "PAP status confirmed as not enforced: $papStatus"
}

# Execute the gcloud command to add the IAM policy binding without condition
Write-Host "Adding IAM policy binding for public access to all objects..."
gcloud storage buckets add-iam-policy-binding $BucketName `
  --member=$Member `
  --role=$Role

# Check the exit code for success or failure
if ($LASTEXITCODE -eq 0) {
    Write-Host "IAM policy binding added successfully."
    # Verify the policy
    $policy = gcloud storage buckets get-iam-policy $BucketName --format=json
    Write-Host "Updated IAM policy: $policy"
} else {
    Write-Host "Failed to add IAM policy binding. Check logs for details."
    # Attempt to describe the policy for debugging
    $policy = gcloud storage buckets get-iam-policy $BucketName --format=json
    Write-Host "Current IAM policy: $policy"
    exit 1
}