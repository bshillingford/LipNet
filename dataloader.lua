--[[

LipNet: End-to-End Sentence-level Lipreading. arXiv preprint arXiv:1611.01599 (2016).

Copyright (C) 2017 Yannis M. Assael, Brendan Shillingford, Shimon Whiteson, Nando de Freitas

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

]]--


local classic = require 'classic'
require 'classic.torch' -- serialization support

require 'image'
require 'hdf5'
require 'nn'
require 'pprint'
require 'csvigo'

local kwargs = require 'util.kwargs'
local log = require 'util.log'


local Threads = require 'threads'
-- Threads.serialization('threads.sharedserialize')

local function trim(s)
    return (s:gsub("^%s*(.-)%s*$", "%1"))
end

function deepcopy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in next, orig, nil do
            copy[deepcopy(orig_key)] = deepcopy(orig_value)
        end
        setmetatable(copy, deepcopy(getmetatable(orig)))
    else -- number, string, boolean, etc
        copy = orig
    end
    return copy
end


function math.round(x)
    return math.floor(x + 0.5)
end


local LipsData = classic.class("LipsData")

function LipsData:_init(opt)
    local opt = kwargs(opt, {
        { 'num_workers', type = 'int-pos', default = 10 },
        { 'datapath', type = 'string' }, -- directory containing numeric video ID subdirectories
        { 'bs', type = 'int-pos', defualt = 20 }, -- batch size
        { 'test_overlapped', type = 'number', default = 0 },
        { 'data_augmentation_temporal', type = 'number', default = 0 },
        { 'normalise', type = 'int', default = 1 },
        { 'frame_rate', type = 'number', default = 25 }, -- fps
        { 'frame_skip', type = 'number', default = 1 }, -- frame skip
        { 'min_timesteps', type = 'int-pos', default = 2 }, -- min frames, for filtering bad data
        { 'max_timesteps', type = 'int-pos', default = 75 }, -- maximum number of frames per sub, for preallocation
        { 'mode_img', type = 'string', default = 'mouth' },
        { 'debug', type = 'int', default = 0 }, -- use host-pinned FloatTensor
        ignore_extras = true
    })

    self.opt = opt
    self.state = {}

    self.overlapped_list = torch.load('util/list_overlapped.t7')
end

-- init function separate from ctor
function LipsData:load_data()
    log.info('entered load_data()...')
    local opt = self.opt

    local vocab_unordered = {}
    vocab_unordered[' '] = true

    self.dataset = {}
    self.dataset_val = {}

    -- Iterate actors
    local count_s = 0
    for dir_s in paths.iterdirs(opt.datapath) do

        count_s = count_s + 1
        xlua.progress(count_s, 33)

        -- Get actor videos
        for dir_v in paths.iterdirs(opt.datapath .. '/' .. dir_s) do
            if opt.debug == 0 or (opt.debug == 1 and (#self.dataset <= opt.bs * 10 or #self.dataset_val <= opt.bs * 10)) then
                local cur_path = opt.datapath .. '/' .. dir_s .. '/' .. dir_v

                -- Load filter
                local flag_add = true

                -- Check if sub was transcribed
                local sub_file = 'data_subs/' .. dir_s .. '/' .. dir_v .. '.align'
                if not path.exists(sub_file) then
                    flag_add = false
                    log.error(string.format('(dir_s == \'%s\', dir_v == \'%s\' - sub) ', dir_s, dir_v))
                end

                -- Check if frames exist
                if not path.exists(cur_path .. '/' .. opt.mode_img) then
                    flag_add = false
                    log.error(string.format('(dir_s == \'%s\', dir_v == \'%s\' - frames) ', dir_s, dir_v))
                end


                local frames = 0
                for _ in paths.iterfiles(cur_path .. '/' .. opt.mode_img) do
                    frames = frames + 1
                end
                if frames ~= 75 then
                    flag_add = false
                    log.error(string.format('(dir_s == \'%s\', dir_v == \'%s\' - frame len) ', dir_s, dir_v))
                end

                -- If filter passed
                if flag_add then

                    -- Load subs
                    local sub_total = {}

                    local line_count = 0
                    for line in io.lines(sub_file) do
                        line_count = line_count + 1
                    end

                    local d = { s = dir_s, v = dir_v, words = {}, t_start = {}, t_end = {} }

                    for line in io.lines(sub_file) do
                        local tok = {}
                        for t in line:gmatch("%w+") do
                            table.insert(tok, t)
                        end

                        -- Remove silence and space
                        if tok[3] ~= 'sil' and tok[3] ~= 'sp' then

                            -- Store sub
                            local sub = tok[3]

                            -- append
                            table.insert(d.words, sub)
                            table.insert(d.t_start, tok[1])
                            table.insert(d.t_end, tok[2])

                            -- Build vocabulary
                            for char in sub:gmatch "." do
                                vocab_unordered[char] = true
                            end
                        end
                    end

                    -- Read image size
                    if self.opt.size == nil then
                        self.opt.size = image.load(cur_path .. '/' .. opt.mode_img .. '/' .. '1.jpg'):size():totable()
                    end

                    -- Append to subs data
                    -- #d.sub > sub_text_limit and d.frames >= opt.min_timesteps and d.frames <= opt.max_timesteps
                    if opt.train_all ~= 1 and ((opt.test_overlapped == 0 and (d.s == 's1' or d.s == 's2' or d.s == 's20' or d.s == 's22')) or (opt.test_overlapped == 1 and self.overlapped_list[dir_s][dir_v] == true)) then
                        if opt.debug == 0 or (opt.debug == 1 and #self.dataset_val <= opt.bs * 2) then
                            d.mode = 7
                            d.flip = 0
                            d.test = 1
                            table.insert(self.dataset_val, d)
                        end
                    else
                        if opt.debug == 0 or (opt.debug == 1 and #self.dataset <= opt.bs * 20) then
                            d.test = 0
                            for flip = 0, 1 do
                                if opt.use_words == 1 then
                                    if opt.test_random == 1 then
                                        local d_i = deepcopy(d)
                                        d_i.flip = flip
                                        d_i.mode = 1
                                        table.insert(self.dataset, d_i)
                                    else
                                        for w_start = 1, 6 do
                                            local d_i = deepcopy(d)
                                            d_i.mode = 1
                                            d_i.w_start = w_start
                                            d_i.flip = flip

                                            d_i.w_end = d_i.w_start + d_i.mode - 1
                                            local frame_v_start = math.max(math.round(75 / 3000 * d.t_start[d_i.w_start]), 1)
                                            local frame_v_end = math.min(math.round(75 / 3000 * d.t_end[d_i.w_end]), 75)
                                            if frame_v_end - frame_v_start + 1 >= 3 then
                                                table.insert(self.dataset, d_i)
                                            end
                                        end
                                    end
                                end

                                local d_i = deepcopy(d)
                                d_i.mode = 7
                                d_i.flip = flip
                                table.insert(self.dataset, d_i)
                            end
                        end
                    end
                end
            end
        end
    end

    -- sort into a table (i.e. keys become 1..N)
    self.vocab = {}
    for char in pairs(vocab_unordered) do self.vocab[#self.vocab + 1] = char end
    table.sort(self.vocab)
    self.opt.vocab_size = #self.vocab

    -- invert ordered to create the char->int mapping
    self.vocab_mapping = {}
    for i, char in ipairs(self.vocab) do
        self.vocab_mapping[char] = i
    end

    log.info(string.format('videos = %d, videos test = %d, vocab = %d', #self.dataset, #self.dataset_val, #self.vocab))

    log.info(string.format('frame size = %d,%d,%d', unpack(self.opt.size)))

    log.info(string.format('vocab = %s', table.concat(self.vocab, "|")))

    -- Create data loader threads
    log.info(string.format('Starting %d data-threads', opt.num_workers))
    self.undergrads = Threads(opt.num_workers,
        function()
            require 'torch'
        end,
        function(idx)
            tid = idx
            torch.manualSeed(tid)
            torch.setnumthreads(1)
            torch.setdefaulttensortype('torch.FloatTensor')

            require('dataloader_t')
            -- paths.dofile('dataloader_t.lua')
            -- print(string.format('Started undergrad with id: %d seed: %d', tid, seed))
        end);

    log.info(string.format('Started %d data-threads', opt.num_workers))
    self.undergrads:addjob(function() return 1 end, function(n) end)
    self.undergrads:synchronize()
    log.info(string.format('Tested %d data-threads', opt.num_workers))
end


-- serialization: (skip self.dataset)
function LipsData:__write(file)
    log.info('serializing a LipsData object')
    file:writeObject({
        opt = self.opt,
        state = self.state,
        -- skip self.dataset: contains stuff like many strings and hdf5 handles
    })
end

function LipsData:__read(file)
    log.info('deserializing a LipsData object')
    local obj = file:readObject()
    self.opt = obj.opt
    self.state = obj.state
end

function LipsData:forward(...)
    return self:updateOutput(...)
end

function LipsData:updateOutput(id, test)
    if not self.dataset then
        log.info('reloading data...')
        self:load_data()
    end

    local opt = self.opt
    local bs
    if id == nil then
        bs = opt.bs
    else
        bs = #id
    end
    local max_time = opt.max_timesteps

    -- TODO: resize self.x depending on the max sampled seq length?

    -- get img height/width:
    local chan, H, W = unpack(self.opt.size)

    -- create the 3 buffers if nonexistent:
    if not self.x then
        -- images: T x bs x chan x H x W
        log.info('using CudaHostTensor')
        self.x = torch.CudaTensor(opt.bs, chan, max_time, H, W)

        -- input lengths: bs-length table of integers, representing the number of input timesteps/frames for the given batch element
        self.len = torch.CudaTensor(opt.bs)
    end

    -- targets: bs-length table of targets (each one is the length of the target seq)
    self.y = {}

    -- pick random videos, load using _read_data(), shove into self.x
    self.x:zero()
    local max_len = 0

    -- Load data in threads
    local opt = self.opt
    local vocab_mapping = self.vocab_mapping

    local data = {}
    for b = 1, bs do

        local d_id, d
        if id == nil then
            if test then
                d_id = torch.random(#self.dataset_val)
            else
                d_id = torch.random(#self.dataset)
            end
        else
            d_id = id[b]
        end

        if test then
            d = self.dataset_val[d_id]
        else
            d = self.dataset[d_id]
        end

        self.undergrads:addjob(function()
            local data_i = read_data(d, opt, vocab_mapping)
            return data_i
        end,
            function(data_i)
                table.insert(data, data_i)
            end)
    end

    -- Sync threads
    self.undergrads:synchronize()
    assert(#data == bs, 'threads out of sync')


    -- Copy to device vectors
    for b = 1, bs do
        -- Fetch x, y
        local x, y = unpack(data[b])


        -- Copy Frames
        self.x[{ { b }, {}, { 1, x:size(2) }, {}, {} }]:copy(x)
        self.len[b] = x:size(2)

        -- Store length
        if self.len[b] > max_len then max_len = self.len[b] end

        -- Store subtitle
        self.y[b] = y
    end

    max_len = math.max(max_len, opt.min_timesteps)

    return {
        self.x:narrow(3, 1, max_len):narrow(1, 1, bs),
        self.y,
        self.len:narrow(1, 1, bs)
    }
end

return LipsData
