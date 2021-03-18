from django.core.exceptions import ValidationError
from django.db import models
from django.utils.translation import ugettext_lazy as _

RESPONSE_KIND_TO_RESPONSES_OPTIONS = {'left_or_right': ['left', 'right', 'tie', 'abstain']}

def validate_inclusion_of_response_kind(value):
    kinds = RESPONSE_KIND_TO_RESPONSES_OPTIONS.keys()
    if value not in kinds:
        raise ValidationError(_('%(value)s is not included in %(kinds)s'), params={'value': value, 'kinds': kinds}, )

class Clip(models.Model):
    created_at = models.DateTimeField('date created', auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True, db_index=True)

    media_url = models.TextField('media url', db_index=True)
    environment_id = models.TextField('environment id', db_index=True)
    clip_tracking_id = models.IntegerField('clip tracking id', db_index=True)

    source = models.TextField('note of where the clip came from', default="", blank=True)

    def __str__(self):
        return "Clip {}".format(self.clip_tracking_id)

class Comparison(models.Model):
    created_at = models.DateTimeField('date created', auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True, db_index=True)

    left_clip = models.ForeignKey(Clip, db_index=True, related_name="compared_on_the_left", on_delete=models.DO_NOTHING)
    right_clip = models.ForeignKey(Clip, db_index=True, related_name="compared_on_the_right", on_delete=models.DO_NOTHING)

    shown_to_tasker_at = models.DateTimeField('time shown to tasker', db_index=True, blank=True, null=True)
    responded_at = models.DateTimeField('time response received', db_index=True, blank=True, null=True)
    response_kind = models.TextField('the response from the tasker', db_index=True,
                                     validators=[validate_inclusion_of_response_kind])
    response = models.TextField('the response from the tasker', db_index=True, blank=True, null=True)
    experiment_name = models.TextField('name of experiment')

    priority = models.FloatField('site will display higher priority items first', db_index=True)
    note = models.TextField('note to be displayed along with the query', default="", blank=True)

    # The Binary Search/Sort Tree that this comparison belongs to. Only used for new-style experiments.
    tree_node = models.ForeignKey('SortTree', null=True, blank=True, default=None, on_delete=models.DO_NOTHING)
    # Whether this comparison is related to a pending clip for said node. Helper used for new-style experiments.
    relevant_to_pending_clip = models.BooleanField(default=False)

    def __str__(self):
        return "Comparison {} ({} vs {})".format(self.id, self.left_clip, self.right_clip)

    # Validation
    def full_clean(self, exclude=None, validate_unique=True):
        super(Comparison, self).full_clean(exclude=exclude, validate_unique=validate_unique)
        self.validate_inclusion_of_response()

    @property
    def response_options(self):
        try:
            return RESPONSE_KIND_TO_RESPONSES_OPTIONS[self.response_kind]
        except KeyError:
            raise KeyError("{} is not a valid response_kind. Valid response_kinds are {}".format(
                self.response_kind, RESPONSE_KIND_TO_RESPONSES_OPTIONS.keys()))

    def validate_inclusion_of_response(self):
        # This can't be a normal validator because it depends on a value
        if self.response is not None and self.response not in self.response_options:
            raise ValidationError(
                _('%(value)s is not included in %(options)s'),
                params={'value': self.response, 'options': self.response_options}, )

class SortTree(models.Model):
    """ Extends a red-black tree to handle async clip sorting with equivalence. """

    parent = models.ForeignKey('self', null=True, related_name='+', on_delete=models.DO_NOTHING)
    left = models.ForeignKey('self', null=True, related_name='+', on_delete=models.DO_NOTHING)
    right = models.ForeignKey('self', null=True, related_name='+', on_delete=models.DO_NOTHING)

    pending_clips = models.ManyToManyField(Clip, related_name='pending_sort_locations')
    bound_clips = models.ManyToManyField(Clip, related_name='tree_bindings')

    experiment_name = models.TextField('name of experiment')

    is_red = models.BooleanField()  # Used for red-black autobalancing

    def __str__(self):
        return "Node {}".format(self.id)

    # I could theoretically do these with a setter decorator,
    # but I want to be able to manipulate them directly without autosaving if needed.
    def make_red(self):
        self.is_red = True
        self.save()

    def make_black(self):
        self.is_red = False
        self.save()

    def set_left(self, x):
        self.left = x
        if x:
            x.parent = self
            x.save()
        self.save()

    def set_right(self, x):
        self.right = x
        if x:
            x.parent = self
            x.save()
        self.save()
