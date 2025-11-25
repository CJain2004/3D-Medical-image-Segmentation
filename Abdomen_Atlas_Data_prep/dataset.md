| File/Folder              | Content                       | Purpose                    |
| ------------------------ | ----------------------------- | -------------------------- |
| `ct.nii.gz`              | CT scan volume                | Model input                |
| `combined_labels.nii.gz` | Multi-class segmentation mask | Multi-class training       |
| `segmentations/`         | Individual binary masks       | Organ-specific training    |
| `BDMAP_xxxxxxxx/`        | One patient’s folder          | Organizes per-patient data |
