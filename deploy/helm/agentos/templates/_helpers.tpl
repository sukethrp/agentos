{{- define "agentos.name" -}}
{{- default .Chart.Name .Values.name -}}
{{- end }}
